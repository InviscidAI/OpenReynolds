from __future__ import annotations

import time
from pathlib import Path

import pytest

from openreynolds.backend.base import ExecResult, JobStatus
from openreynolds.backend.base import WORKSPACE_ROOT
from openreynolds.tools import TOOLS, ToolContext, dispatch


def test_tool_list_is_deterministic():
    """Tool order is prefix position 0 for the cache, so it stays sorted and fixed."""
    names = [tool["name"] for tool in TOOLS]
    assert names == sorted(names)
    assert names == [
        "bash",
        "cad",
        "fetch",
        "job_check",
        "job_kill",
        "job_start",
        "mesh_review",
        "read_file",
        "write_file",
    ]
    for tool in TOOLS:
        assert tool["input_schema"]["type"] == "object"
        assert tool["description"]


def test_the_cad_tool_says_what_comes_back_and_what_does_not():
    """The description says what the tool returns and, just as importantly, what it
    does not: a result read as "the case is ready to solve" is the failure this tool
    was rebuilt to stop. The restraint test_prompt applies to the main prompt applies
    here too -- the clause says what comes back, never what the caller should do."""
    description = next(t for t in TOOLS if t["name"] == "cad")["description"]
    assert "checkMesh" in description and "patch table" in description
    assert "no boundary conditions" in description and "no solve" in description
    lowered = description.lower()
    for imperative in ("you must", "always ", "never ", "you should", "prefer "):
        assert imperative not in lowered, imperative


def test_the_checkpoint_tool_is_offered_only_in_structured_mode(ctx):
    """Nine tools in every mode; a tenth, `checkpoint`, only when the person chose
    structured mode. Still sorted, so a mode's tool list is always the same bytes."""
    from openreynolds.tools import tools_for

    ctx.cad = object()
    ctx.reviewer = object()
    for mode in ("auto", "partial"):
        ctx.mode = mode
        assert "checkpoint" not in [tool["name"] for tool in tools_for(ctx)]
    ctx.mode = "structured"
    names = [tool["name"] for tool in tools_for(ctx)]
    assert names == sorted(names)
    assert names == sorted([tool["name"] for tool in TOOLS] + ["checkpoint"])
    assert tools_for(ctx) == tools_for(ctx)


def test_a_checkpoint_outside_structured_mode_asks_nobody(ctx):
    asked = []
    ctx.approver = type("A", (), {"ask": lambda self, *a: asked.append(a)})()
    content, is_error = dispatch(ctx, "checkpoint", {"stage": "plan", "summary": "s", "next": "n"})
    assert not asked and "not put to the person" in content


def test_a_checkpoint_approved_with_all_switches_to_full_auto(ctx):
    from openreynolds.approval import Decision

    switched = []
    ctx.mode = "structured"
    ctx.on_mode = switched.append
    ctx.approver = type("A", (), {"ask": lambda self, *a: Decision(True, all=True)})()
    content, is_error = dispatch(ctx, "checkpoint", {"stage": "mesh", "summary": "s", "next": "solve"})
    assert not is_error and ctx.plan_approved
    assert switched == ["auto"]
    assert "solve" in content and "full auto" in content


def test_long_job_description_keeps_output_in_the_job_log():
    """Redirecting a solver to log.simpleFoam left job_check with zero bytes even
    though the solve was healthy. Detached jobs already capture stdout and stderr."""
    description = next(t for t in TOOLS if t["name"] == "job_start")["description"]
    assert "Do not redirect" in description
    assert "job_check" in description
    assert "tee" in description


def test_job_start_refuses_stdio_redirects(ctx, backend):
    """C-03/C-07/C-08 hid the solver log from job_check by redirecting into a file."""
    content, is_error = dispatch(
        ctx, "job_start", {"cmd": "mpirun -np 4 interFoam -parallel > log.interFoam 2>&1"}
    )
    assert not backend.started
    assert content.startswith("not started:")
    assert "redirect" in content.lower()
    content, _ = dispatch(ctx, "job_start", {"cmd": "simpleFoam | tee log.simpleFoam"})
    assert not backend.started
    assert "tee" in content.lower()
    content, _ = dispatch(ctx, "job_start", {"cmd": "simpleFoam"})
    assert backend.started and content.startswith("started job")


def test_environment_warns_that_partial_openfoam_banners_are_invalid():
    """A generated controlDict opened the decorative C++ banner but never closed it,
    so OpenFOAM treated the whole dictionary as a comment and lost a minute diagnosing
    a format issue. The minimal safe file starts directly with FoamFile."""
    environment = (
        Path(__file__).parents[1] / "openreynolds" / "toolbox" / "ENVIRONMENT.md"
    ).read_text(encoding="utf-8")
    assert "start directly with `FoamFile`" in environment
    assert "partial decorative banner" in environment


def test_the_cad_tool_is_offered_only_when_there_is_a_desk_behind_it(ctx):
    """A tool in the list that can only answer "not available" costs the model a call
    to find that out. Taking it out is also what makes the question answerable: the
    same prompt run with the desk and without it is the only honest way to settle
    whether a slow natural-language sub-agent beats the bash the caller already has
    (`OPENREYNOLDS_CAD_TOOL=0`, or `OPENREYNOLDS_MESH_TOOL=0` as it was)."""
    from openreynolds.tools import tools_for

    assert "cad" not in [tool["name"] for tool in tools_for(ctx)]
    ctx.cad = object()
    assert "cad" in [tool["name"] for tool in tools_for(ctx)]
    # With every conditional tool wired the list is `TOOLS` itself, the same object
    # each call, so the prefix the cache is keyed on never moves between requests.
    ctx.reviewer = object()
    assert tools_for(ctx) is TOOLS
    assert tools_for(ctx) is TOOLS


def test_the_mesh_review_tool_is_offered_only_when_there_is_a_reviewer_behind_it(ctx):
    """The same rule as `cad`, for the same reason: a reviewer is a model call with
    pictures in it, and without one the tool could only say so. Each of the two is
    withheld on its own, so a session with a desk and no reviewer -- or the reverse --
    offers exactly what it can serve."""
    from openreynolds.tools import tools_for

    def names():
        return [tool["name"] for tool in tools_for(ctx)]

    assert "mesh_review" not in names() and "cad" not in names()
    ctx.reviewer = object()
    assert "mesh_review" in names() and "cad" not in names()
    assert names() == sorted(names())
    ctx.reviewer = None
    ctx.cad = object()
    assert "cad" in names() and "mesh_review" not in names()


def test_the_mesh_review_description_says_what_it_is_not():
    """Read on its own by a model deciding whether to call it, the description has to
    carry the two facts a reviewer is most easily mistaken about: it is not checkMesh,
    and it judges the shape rather than the mesh. And it is a statement of what comes
    back, in the register every tool description here keeps -- no instruction."""
    import re

    from test_prompt import IMPERATIVE_PATTERNS

    tool = next(t for t in TOOLS if t["name"] == "mesh_review")
    description = tool["description"]
    assert "does not run checkMesh" in description
    assert "Not a substitute for checkMesh" in description
    assert "PASS or FAIL" in description and "picture" in description
    for pattern in IMPERATIVE_PATTERNS:
        assert not re.search(pattern, description, re.IGNORECASE), pattern
    schema = tool["input_schema"]
    assert set(schema["properties"]) == {"case", "request", "notes"}
    assert schema["required"] == ["case", "request"]
    assert "/work" in schema["properties"]["case"]["description"]


# -- mesh_review: the reviewer reached without the desk ---------------------------


class ScriptedReviewer:
    """A `cad.Reviewer` that answers with the review it was handed and remembers what
    it was asked. It never renders and never calls a model: the handler under test is
    the plumbing round it, not the review itself (`tests/test_cad_review.py`)."""

    def __init__(self, review):
        self.review_ = review
        self.calls: list[dict] = []

    def review(self, case_dir, case_rel, request, said, summary, facts, prior=None):
        self.calls.append(dict(case_dir=case_dir, case_rel=case_rel, request=request,
                               said=said, summary=summary, facts=facts, prior=prior))
        return self.review_


def _passing_review(png=b"\x89PNG-first-view", tokens=None):
    from openreynolds.cad.review import Review

    return Review(verdict="pass", confidence="sure", views=["renders/review/overview.png",
                                                             "renders/review/cells.png",
                                                             "renders/review/detail.png"],
                  png=png, tokens=tokens if tokens is not None else {"input_tokens": 3000,
                                                                      "output_tokens": 200})


def _failing_review(png=b"\x89PNG-first-view"):
    from openreynolds.cad.review import Problem, Review

    return Review(verdict="fail", confidence="likely", views=["iso.png", "ortho.png", "cuts.png"],
                  png=png, problems=[Problem(
                      what="the lobes are polygons, not curves",
                      seen_in="detail.png, top-left quadrant",
                      why_it_matters="the flow separates at each corner", severity="blocking")])


def _a_case(backend, case="/work/study-test/mesh"):
    backend.dirs[case] = ["constant", "system", "Allmesh"]
    backend.dirs[f"{case}/constant/polyMesh"] = ["points", "faces", "owner", "neighbour", "boundary"]
    return case


def test_without_a_reviewer_the_tool_says_so_rather_than_pretending(ctx):
    answer, is_error = dispatch(ctx, "mesh_review", {"case": "mesh", "request": "a duct"})
    assert not is_error
    assert "not available" in answer and "mesh_look.py" in answer


def test_a_review_comes_back_as_the_first_view_and_the_verdict(ctx, backend):
    """The picture first and the words second, the shape every image result here has:
    when the picture is evicted from the thread the caption still carries the verdict.
    The tokens the reviewer spent join the session's totals through the same hook the
    desk's do, and the case is handed over as the workspace path with its study name."""
    ctx.home = "/work/study-test"
    case = _a_case(backend)
    ctx.reviewer = ScriptedReviewer(_passing_review())
    spent = []
    ctx.on_tokens = spent.append

    content, is_error = dispatch(ctx, "mesh_review", {
        "case": "mesh", "request": "a 2D Tesla valve, 40 mm long, inlet on the left",
        "notes": "the desk says 38.5 mm along x"})

    assert not is_error
    assert isinstance(content, list) and content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    text = content[1]["text"]
    assert text.startswith("independent review: PASS -- looked at 3 views")
    assert "no checkMesh ran here and nothing about the mesh was changed" in text
    assert "fix the shape" not in text, "a pass has nothing to fix"
    assert spent == [{"input_tokens": 3000, "output_tokens": 200}]
    call = ctx.reviewer.calls[0]
    assert call["case_dir"] == case and call["case_rel"] == "mesh"
    assert call["request"].startswith("a 2D Tesla valve")
    assert call["summary"] == "the desk says 38.5 mm along x"
    assert call["said"] is None and call["facts"] is None and call["prior"] is None


def test_a_failing_review_names_the_problems_and_what_to_do_next(ctx, backend):
    ctx.home = "/work/study-test"
    _a_case(backend)
    ctx.reviewer = ScriptedReviewer(_failing_review())

    content, is_error = dispatch(ctx, "mesh_review", {"case": "mesh", "request": "a Tesla valve"})

    assert not is_error
    text = content[1]["text"]
    assert text.startswith("independent review: FAIL -- looked at 3 views:")
    assert "the lobes are polygons, not curves (seen in detail.png, top-left quadrant)" in text
    assert "fix the shape and call this again, or hand the problems to the cad tool" in text


def test_a_review_with_no_picture_is_words_alone(ctx, backend):
    """A render that could not be fetched is a `skipped` review with no PNG; the tool
    result is then text, never an image block with nothing in it."""
    from openreynolds.cad.review import Review

    ctx.home = "/work/study-test"
    _a_case(backend)
    ctx.reviewer = ScriptedReviewer(Review(verdict="skipped", error="could not render the mesh for review: no pyvista"))

    content, is_error = dispatch(ctx, "mesh_review", {"case": "mesh", "request": "a duct"})

    assert not is_error and isinstance(content, str)
    assert content.startswith("not reviewed: could not render the mesh for review")
    assert "nothing about the mesh was changed" in content


def test_an_absolute_case_under_the_workspace_is_taken_as_it_is(ctx, backend):
    ctx.home = "/work/study-test"
    case = _a_case(backend, "/work/other-study/run")
    ctx.reviewer = ScriptedReviewer(_passing_review(png=None))

    content, _ = dispatch(ctx, "mesh_review", {"case": case, "request": "a duct"})

    assert isinstance(content, str) and content.startswith("independent review: PASS")
    call = ctx.reviewer.calls[0]
    assert call["case_dir"] == case
    assert call["case_rel"] == case, "outside this study, the name is the whole path"


@pytest.mark.parametrize("case", [
    "/home/someone/case",
    "../../etc",
    "/work/../home/someone",
])
def test_a_case_outside_the_workspace_is_refused_before_anything_is_spent(ctx, backend, case):
    """`refuse_input`'s rule, applied where the path lands rather than how it is spelled:
    a `..` that climbs out of the root is the same refusal as an absolute path outside
    it. Nothing is rendered and nothing is asked of a model for a path like that."""
    ctx.home = "/work/study-test"
    ctx.reviewer = ScriptedReviewer(_passing_review())

    answer, is_error = dispatch(ctx, "mesh_review", {"case": case, "request": "a duct"})

    assert not is_error
    assert answer.startswith("nothing was reviewed:")
    assert "not under /work/" in answer
    assert ctx.reviewer.calls == [] and backend.execs == []


def test_a_case_that_is_not_there_costs_a_stat_and_a_sentence(ctx, backend):
    ctx.home = "/work/study-test"
    ctx.reviewer = ScriptedReviewer(_passing_review())

    answer, is_error = dispatch(ctx, "mesh_review", {"case": "absent", "request": "a duct"})

    assert not is_error
    assert answer.startswith("nothing was reviewed: /work/study-test/absent could not be read")
    assert "not_found" in answer
    assert ctx.reviewer.calls == []


def test_a_case_with_no_polymesh_is_refused_without_a_render(ctx, backend):
    ctx.home = "/work/study-test"
    backend.dirs["/work/study-test/empty"] = ["system"]
    ctx.reviewer = ScriptedReviewer(_passing_review())

    answer, _ = dispatch(ctx, "mesh_review", {"case": "empty", "request": "a duct"})

    assert answer.startswith("nothing was reviewed:")
    assert "no constant/polyMesh" in answer
    assert ctx.reviewer.calls == []


def test_a_review_needs_a_request_and_a_case(ctx, backend):
    """The reviewer compares the pictures to the request; without one there is nothing
    to compare against, and a review of "whatever this is" would pass anything."""
    ctx.home = "/work/study-test"
    _a_case(backend)
    ctx.reviewer = ScriptedReviewer(_passing_review())

    answer, _ = dispatch(ctx, "mesh_review", {"case": "mesh", "request": "  "})
    assert answer.startswith("nothing was reviewed:") and "`request`" in answer
    answer, _ = dispatch(ctx, "mesh_review", {"case": "", "request": "a duct"})
    assert answer.startswith("nothing was reviewed:") and "`case`" in answer
    assert ctx.reviewer.calls == []


# -- the desk's answer carries the review -------------------------------------------


def _desk_result(**fields):
    """A `CadResult` as `test_cad_wiring.py` fakes one, with whatever this test sets."""
    base = {"tokens": {}, "png": None, "check": None, "ok": True, "error": "",
            "case_rel": "mesh", "summary": "", "steps": [], "seconds": 60.0,
            "stopped": "", "remarks": []}
    base.update(fields)
    return type("R", (), base)()


def test_cad_text_puts_the_review_right_after_the_state_of_the_mesh(ctx):
    """First the state of `constant/polyMesh`, as the docstring demands; then the one
    judgement about the shape. A mesh that passed checkMesh and did not pass review is
    the case the reviewer exists for, and it has to be read before the desk's own
    account of what it built."""
    from openreynolds.tools import cad_text

    review = _failing_review()
    review.unresolved = True
    review.round = 2
    text = cad_text(_desk_result(review=review, summary="built and measured, 40 mm"))
    lines = text.splitlines()
    assert lines[0] == "nothing was meshed in mesh", "no check: the first line says so"
    assert lines[1] == ("independent review: did not pass after 2 rounds -- treat this "
                        "mesh as suspect:")
    assert "the lobes are polygons" in lines[2]
    assert text.index("independent review") < text.index("the CAD desk says:")
    assert "call mesh_review to have it looked at again independently" in text


def test_cad_text_says_when_a_finished_mesh_was_never_reviewed(ctx):
    """The end-of-run check accepts what is there when the desk runs out of steps or
    time, and nobody looked at the shape. A missing line would read as a clean bill."""
    from openreynolds.tools import NOT_REVIEWED_BUDGET, cad_text

    for stopped in ("steps", "time", "provider"):
        text = cad_text(_desk_result(stopped=stopped))
        assert NOT_REVIEWED_BUDGET in text, stopped
    assert NOT_REVIEWED_BUDGET not in cad_text(_desk_result(stopped=""))
    assert NOT_REVIEWED_BUDGET not in cad_text(_desk_result(ok=False, stopped="steps"))
    assert NOT_REVIEWED_BUDGET not in cad_text(_desk_result(stopped="steps",
                                                            review=_passing_review()))


def test_the_cad_tool_falls_back_to_the_reviewers_first_view(ctx):
    """`CoreDesk` draws nothing at its finish, so without this the caller of the
    measured desk got words about a shape and no shape. The reviewer's first view is
    a picture of exactly that mesh, and the desk's own PNG still wins when there is one."""

    class Desk:
        def __init__(self, result):
            self.result = result

        def run(self, request, case=None, geometry="", inputs=()):
            return self.result

    ctx.cad = Desk(_desk_result(review=_passing_review(png=b"\x89PNG-review")))
    content, is_error = dispatch(ctx, "cad", {"request": "a duct"})
    assert not is_error and isinstance(content, list)
    import base64
    assert base64.b64decode(content[0]["source"]["data"]) == b"\x89PNG-review"
    assert "independent review: PASS -- looked at 3 views" in content[1]["text"]

    ctx.cad = Desk(_desk_result(png=b"\x89PNG-desk", review=_passing_review(png=b"\x89PNG-review")))
    content, _ = dispatch(ctx, "cad", {"request": "a duct"})
    assert base64.b64decode(content[0]["source"]["data"]) == b"\x89PNG-desk"

    ctx.cad = Desk(_desk_result(review=_passing_review(png=None)))
    content, _ = dispatch(ctx, "cad", {"request": "a duct"})
    assert isinstance(content, str), "no picture anywhere: words alone"


def test_reading_a_file_is_bounded_in_time(ctx, backend):
    """Until 2026-09-07 a read had no timeout at all -- one `stat` was watched sitting
    for over ten minutes -- and the backend's own default is five. A contended read is
    usually very slow or fast, so a minute is the honest wait: the model can read again,
    and a tool call it cannot escape is the expensive part."""
    from openreynolds.tools import READ_ATTEMPTS, READ_TIMEOUT_S

    backend.files["/work/x.txt"] = b"hello"
    dispatch(ctx, "read_file", {"path": "/work/x.txt"})
    assert backend.stat_kwargs == {"timeout": READ_TIMEOUT_S, "max_attempts": READ_ATTEMPTS}
    assert backend.get_file_kwargs["timeout"] == READ_TIMEOUT_S


def test_unknown_tool_is_an_error_not_a_crash(ctx):
    content, is_error = dispatch(ctx, "run_gate", {})
    assert is_error
    assert "No such tool" in content


def test_bash_reports_exit_code(ctx, backend):
    backend.exec_result = ExecResult(3, "boom", False, None)
    content, is_error = dispatch(ctx, "bash", {"cmd": "false"})
    assert not is_error
    assert "exit_code: 3" in content
    assert "boom" in content


def test_bash_surfaces_the_workspace_stderr_when_the_wrapper_failed(ctx, backend):
    """A failure of the workspace itself -- a gone working directory, a capture-sync
    that failed -- comes back on the wrapper's stderr, not the command output. It used
    to be dropped, so a platform failure read like a command that produced nothing and
    got retried into the same wall. Now it is shown, labelled as the workspace."""
    backend.exec_result = ExecResult(3, "", False, None, stderr="cwd not found: /work/gone")
    content, is_error = dispatch(ctx, "bash", {"cmd": "echo test"})
    assert not is_error
    assert "workspace:" in content
    assert "cwd not found: /work/gone" in content


def test_bash_says_nothing_extra_when_the_wrapper_stderr_is_empty(ctx, backend):
    backend.exec_result = ExecResult(0, "ok", False, None)
    content, _ = dispatch(ctx, "bash", {"cmd": "echo ok"})
    assert "workspace:" not in content


def test_bash_clamps_and_passes_through_timeout(ctx, backend):
    dispatch(ctx, "bash", {"cmd": "sleep 1", "timeout_s": 42, "cwd": "/work/case"})
    assert backend.last_exec == ("sleep 1", "/work/case", 42)


def test_truncation_marker_points_at_the_tail(ctx, backend):
    """The service returns the head of a long log, so the marker offers the far end."""
    backend.exec_result = ExecResult(0, "H" * 5000, True, "/work/.foamd/exec/abc.log")
    backend.files["/work/.foamd/exec/abc.log"] = b"L" * 1_000_000

    content, is_error = dispatch(ctx, "bash", {"cmd": "solve"})

    assert not is_error
    assert "[truncated" in content
    assert "of 1000000 shown" in content
    assert "/work/.foamd/exec/abc.log" in content
    assert "offset=996000" in content  # 1_000_000 - TAIL_HINT_BYTES
    # our own cap applied on top of the service's
    assert len(content.encode()) < 5000


def test_truncation_marker_without_a_total(ctx, backend):
    backend.exec_result = ExecResult(0, "H" * 5000, True, "/work/.foamd/exec/abc.log")
    content, _ = dispatch(ctx, "bash", {"cmd": "solve"})
    assert "read_file with an offset" in content


def test_our_cap_alone_still_marks(ctx, backend):
    backend.exec_result = ExecResult(0, "x" * 4000, False, None)
    content, _ = dispatch(ctx, "bash", {"cmd": "cat big"})
    assert "[truncated" in content


def test_write_then_read_round_trip(ctx, backend):
    dispatch(ctx, "write_file", {"path": "/work/a.txt", "content": "hello"})
    assert backend.files["/work/a.txt"] == b"hello"

    content, is_error = dispatch(ctx, "read_file", {"path": "/work/a.txt"})
    assert not is_error
    assert "bytes 0–5 of 5" in content
    assert content.endswith("hello")


def test_read_file_window_reports_what_remains(ctx, backend):
    backend.files["/work/big"] = b"0123456789"
    content, _ = dispatch(ctx, "read_file", {"path": "/work/big", "offset": 2, "limit": 3})
    assert "bytes 2–5 of 10" in content
    assert "5 bytes remain past this window" in content
    assert content.endswith("234")


class _StillWriting:
    """A solver appending to its output between the calls a read makes.

    The reads are the FakeBackend's own; only the writer is new. A chunk lands after
    each fetch, which is what a `postProcessing` forces file or a solver log does while
    the case is running."""

    def __init__(self, backend, path, chunks):
        self.backend, self.path, self.chunks = backend, path, list(chunks)
        self.inner = backend.get_file
        self.reads = 0

    def __call__(self, path, *args, **kwargs):
        data = self.inner(path, *args, **kwargs)
        if path == self.path:
            self.reads += 1
            if self.chunks:
                self.backend.files[path] += self.chunks.pop(0)
        return data


def test_a_file_written_under_the_read_is_read_again(ctx, backend):
    """`_read_file` stats the path, asks for that many bytes and is handed exactly that
    many, so the short-read guard that covers images cannot fire on a `.dat`, a `.csv`
    or a log: a file caught mid-write comes back looking whole, and a forces file cut
    short is a number the model will happily average (F-64). A second stat catches it,
    and one more read lands on the finished file."""
    backend.files["/work/forces.dat"] = b"# Time Cd\n0.1 1.0\n"
    backend.get_file = _StillWriting(backend, "/work/forces.dat", [b"0.2 1.1\n"])

    content, is_error = dispatch(ctx, "read_file", {"path": "/work/forces.dat"})

    assert not is_error
    assert backend.get_file.reads == 2, "the first read was taken mid-write"
    assert content.endswith("0.2 1.1\n"), "the whole file, not the part that existed"
    assert "bytes 0–26 of 26" in content
    assert "Still being written" not in content, "it settled; there is nothing to warn about"


def test_a_file_that_never_settles_says_so_rather_than_looking_whole(ctx, backend):
    """A read again is not always enough -- a solver writing every timestep is still
    writing on the second look. Nothing here can make it stop, so what comes back names
    the three sizes and says the snapshot may stop part-way through a record. The
    alternative is a truncated column that reads exactly like a complete one."""
    backend.files["/work/live.dat"] = b"1\n"
    backend.get_file = _StillWriting(backend, "/work/live.dat", [b"2\n", b"3\n", b"4\n"])

    content, is_error = dispatch(ctx, "read_file", {"path": "/work/live.dat"})

    assert not is_error, "a growing file is a fact about the file, not a failed call"
    assert backend.get_file.reads == 2, "read again once, not until it settles"
    assert "Still being written: 2 bytes when it was measured, 4 after the first read, " \
           "6 after the second" in content
    assert "part-way through a line or a record" in content


def test_a_settled_file_costs_one_extra_round_trip_and_no_words(ctx, backend):
    """The check is a stat on a path that already makes several calls, and the common
    case -- a file nothing is writing to -- pays for that stat and nothing else."""
    backend.files["/work/quiet.txt"] = b"hello"
    stats, reads = [], []
    inner_stat, inner_get = backend.stat, backend.get_file
    backend.stat = lambda path, **kw: (stats.append(path), inner_stat(path, **kw))[1]
    backend.get_file = lambda path, *a, **kw: (reads.append(path), inner_get(path, *a, **kw))[1]

    content, is_error = dispatch(ctx, "read_file", {"path": "/work/quiet.txt"})

    assert not is_error
    assert stats == ["/work/quiet.txt"] * 2 and reads == ["/work/quiet.txt"]
    assert "bytes 0–5 of 5" in content and content.endswith("hello")


def test_a_file_that_disappears_under_the_check_still_hands_back_what_was_read(ctx, backend):
    """The bytes are in hand and they were real when they were fetched. A path removed
    by whatever was writing it -- a script clearing its own scratch -- must not turn a
    successful read into a failed tool call: that trades a rare inaccuracy for a common
    one."""
    from openreynolds.backend.base import BackendError

    backend.files["/work/temp.csv"] = b"a,b\n1,2\n"
    inner = backend.stat
    calls = []

    def vanishing(path, **kwargs):
        calls.append(path)
        if len(calls) > 1:
            raise BackendError(f"no such path: {path}", code="not_found", status=404)
        return inner(path, **kwargs)

    backend.stat = vanishing
    content, is_error = dispatch(ctx, "read_file", {"path": "/work/temp.csv"})

    assert not is_error
    assert len(calls) == 2, "the check ran and was refused, which is the case under test"
    assert content.endswith("a,b\n1,2\n")
    assert "Still being written" not in content, "a check that cannot run makes no claim"


def test_read_file_on_a_directory_lists_it(ctx, backend):
    backend.dirs["/work/case"] = ["0", "constant", "system"]
    content, is_error = dispatch(ctx, "read_file", {"path": "/work/case"})
    assert not is_error
    assert "directory, 3 entries" in content
    assert "constant" in content


def test_missing_path_is_a_fact_not_a_traceback(ctx):
    content, is_error = dispatch(ctx, "read_file", {"path": "/work/nope"})
    assert is_error
    assert "not_found" in content


def test_job_start_records_it_locally(ctx, backend, store):
    """The service has no list-jobs endpoint, so a resume depends on this record."""
    content, is_error = dispatch(
        ctx, "job_start", {"cmd": "simpleFoam", "name": "solve", "kill_on": ["FOAM FATAL"]}
    )
    assert not is_error
    job_id = backend.started and "job-1"
    assert job_id in store.session.jobs
    record = store.session.jobs[job_id]
    assert record.name == "solve"
    assert record.cmd == "simpleFoam"
    assert backend.started[0]["kill_on"] == ["FOAM FATAL"]


def test_job_start_refuses_a_kill_pattern_that_matches_the_trapfpe_banner(ctx, backend):
    content, _ = dispatch(
        ctx, "job_start", {"cmd": "simpleFoam", "kill_on": ["Floating point exception"]}
    )

    assert not backend.started
    assert content.startswith("not started:")
    assert "trapFpe" in content
    assert "trapping enabled" in content


RESTARTING_DICT = b"""\
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application     pimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1.3;
deltaT          1e-4;
writeControl    adjustableRunTime;
writeInterval   0.0025;
"""

SOLVE = "cd /work/s/run && mpirun -np 4 pimpleFoam -parallel"
REDIRECTED_SOLVE = SOLVE + " > log.pimpleFoam 2>&1"
LISTING = ("ls -d /work/s/run/[0-9]* /work/s/run/processor*/[0-9]* "
           "/work/s/run/processors*/[0-9]* 2>/dev/null")


def _with_times(backend, dict_text=RESTARTING_DICT, times=("0", "0.2", "0.4", "0.6", "0.8")):
    backend.files["/work/s/run/system/controlDict"] = dict_text
    backend.exec_results[LISTING] = ExecResult(
        0, "\n".join(f"/work/s/run/processors4/{t}" for t in times) + "\n", False, None)


def test_a_solver_relaunch_that_would_overwrite_a_transient_is_refused(ctx, backend, store):
    """The loss this guards, measured: 22 minutes of a transient rewritten from t=0
    because the controlDict said startFrom startTime and the command was launched
    again in the same directory (study 20260920-161908-c7ef)."""
    _with_times(backend)
    content, is_error = dispatch(ctx, "job_start", {"cmd": SOLVE})
    assert not backend.started, "nothing was launched"
    assert content.startswith("not started:")
    assert "4 written time step(s), from 0.2 to 0.8" in content, "the 0 directory is the start, not a write"
    assert "startFrom latestTime" in content and "overwrite=true" in content


def test_start_from_latest_time_carries_a_transient_on(ctx, backend):
    _with_times(backend, RESTARTING_DICT.replace(b"startFrom       startTime;", b"startFrom       latestTime;"))
    content, _ = dispatch(ctx, "job_start", {"cmd": SOLVE})
    assert backend.started and content.startswith("started job")
    assert "startFrom latestTime" in content


def test_overwrite_true_is_the_deliberate_restart(ctx, backend):
    _with_times(backend)
    content, _ = dispatch(ctx, "job_start", {"cmd": SOLVE, "overwrite": True})
    assert backend.started and content.startswith("started job")


def test_a_first_launch_and_a_mesher_are_never_guarded(ctx, backend):
    backend.files["/work/s/run/system/controlDict"] = RESTARTING_DICT
    backend.exec_results[LISTING] = ExecResult(0, "/work/s/run/0\n", False, None)
    content, _ = dispatch(ctx, "job_start", {"cmd": SOLVE})
    assert content.startswith("started job"), "only the initial time exists: not a restart"
    _with_times(backend)
    content, _ = dispatch(ctx, "job_start", {"cmd": "cd /work/s/run && blockMesh"})
    assert content.startswith("started job"), "a mesher writes no time steps"


def test_purge_write_is_said_at_launch(ctx, backend):
    backend.files["/work/s/run/system/controlDict"] = RESTARTING_DICT + b"purgeWrite      2;\n"
    backend.exec_results[LISTING] = ExecResult(0, "", False, None)
    content, _ = dispatch(ctx, "job_start", {"cmd": SOLVE})
    assert "purgeWrite 2 (only the last 2 write times are kept on disk)" in content


def test_a_steady_solver_is_launched_with_the_shape_of_its_residuals(ctx, backend):
    """A steady solver on an unsteady flow levels off, and a model not told to expect
    it read the plateau as a failed run in four studies of five (`convergence`). The
    clause rides on the launch line because the residuals are the next thing read."""
    from openreynolds import convergence

    backend.files["/work/s/run/system/controlDict"] = RESTARTING_DICT.replace(b"pimpleFoam", b"simpleFoam")
    backend.exec_results[LISTING] = ExecResult(0, "", False, None)
    content, _ = dispatch(ctx, "job_start", {"cmd": SOLVE.replace("pimpleFoam", "simpleFoam")})
    assert content.startswith("started job")
    assert f"[{convergence.STEADY_LAUNCH_NOTE}]" in content
    assert content.index("endTime 1.3") < content.index("steady solver:"), "the run's shape first, the reading after"


def test_a_transient_solver_is_launched_without_the_steady_clause(ctx, backend):
    """A transient's per-step residuals do not have that shape, and a note that
    appears only sometimes is a note worth reading."""
    backend.files["/work/s/run/system/controlDict"] = RESTARTING_DICT
    backend.exec_results[LISTING] = ExecResult(0, "", False, None)
    content, _ = dispatch(ctx, "job_start", {"cmd": SOLVE})
    assert "steady solver" not in content
    content, _ = dispatch(ctx, "job_start", {"cmd": "cd /work/s/run && blockMesh"})
    assert "steady solver" not in content, "a mesher is not a solve"


def test_the_steady_clause_survives_a_case_with_no_controldict(ctx, backend):
    """The shape note is silent without a controlDict; the steady clause is about the
    solver, not the dictionary, and is said either way."""
    backend.exec_results[LISTING] = ExecResult(0, "", False, None)
    content, _ = dispatch(ctx, "job_start", {"cmd": "cd /work/s/run && simpleFoam"})
    assert content.startswith("started job") and "steady solver:" in content
    assert "endTime" not in content


def test_the_launch_note_tells_threads_from_cores(ctx, backend):
    """Told "8 cores", a live agent decomposed for 6 and Open MPI refused the run: its
    slots are the physical cores. Both numbers are said, and the one mpirun accepts."""
    backend.files["/work/s/run/system/controlDict"] = RESTARTING_DICT
    backend.exec_results[LISTING] = ExecResult(0, "", False, None)
    content, _ = dispatch(ctx, "job_start", {"cmd": SOLVE.replace("-np 4", "-np 6")})
    assert "8 hardware threads = 4 physical cores" in content
    assert "up to 4 ranks" in content and "6 ranks is more than the 4 cores" in content


def test_job_check_advances_the_offset(ctx, backend, store):
    dispatch(ctx, "job_start", {"cmd": "simpleFoam"})
    backend.logs["job-1"] = b"line one\nline two\n"

    first, _ = dispatch(ctx, "job_check", {"job_id": "job-1"})
    assert "line one" in first
    assert store.session.jobs["job-1"].log_offset == 18

    backend.logs["job-1"] += b"line three\n"
    second, _ = dispatch(ctx, "job_check", {"job_id": "job-1"})
    assert "line three" in second
    assert "line one" not in second


def test_job_check_surfaces_the_kill_on_line(ctx, backend):
    dispatch(ctx, "job_start", {"cmd": "simpleFoam", "kill_on": ["FOAM FATAL"]})
    backend.jobs["job-1"] = JobStatus(
        job_id="job-1",
        status="killed",
        end_reason="kill_on_match",
        killed_by="--> FOAM FATAL ERROR: keyword nu is undefined",
    )
    content, _ = dispatch(ctx, "job_check", {"job_id": "job-1"})
    assert "end_reason=kill_on_match" in content
    assert "keyword nu is undefined" in content


def test_job_kill_updates_the_record(ctx, store):
    dispatch(ctx, "job_start", {"cmd": "simpleFoam"})
    content, _ = dispatch(ctx, "job_kill", {"job_id": "job-1"})
    assert "status=killed" in content
    assert store.session.jobs["job-1"].end_reason == "killed_by_client"


def test_fetch_writes_locally_and_notifies(ctx, backend, store):
    seen: list = []
    ctx.on_fetch = seen.extend
    backend.files["/work/case/renders/u.png"] = b"\x89PNG"

    content, is_error = dispatch(ctx, "fetch", {"paths": ["/work/case/renders/u.png"]})

    assert not is_error
    assert "copied 1 file" in content
    # The workspace shape is preserved, so two cases' renders cannot collide.
    assert (store.fetch_dir() / "case" / "renders" / "u.png").read_bytes() == b"\x89PNG"
    assert len(seen) == 1


def test_fetch_of_nothing_is_not_an_error(ctx):
    content, is_error = dispatch(ctx, "fetch", {"paths": []})
    assert not is_error
    assert "nothing was copied" in content


def test_job_start_output_is_parseable_by_the_smoke_script(ctx):
    """scripts/smoke.py reads the id out of this line; pin the shape."""
    content, _ = dispatch(ctx, "job_start", {"cmd": "simpleFoam", "name": "ticker"})
    assert content.split()[:2] == ["started", "job"]
    assert content.split()[2] == "job-1"


def test_an_over_long_timeout_is_reported_not_silently_clamped(ctx, backend):
    """A command cut off at a ceiling the caller did not know about reads as one
    that finished."""
    content, _ = dispatch(ctx, "bash", {"cmd": "simpleFoam", "timeout_s": 600})
    assert "exceeds the 300s ceiling" in content
    assert "job_start has no such limit" in content


def test_an_ordinary_timeout_says_nothing(ctx):
    content, _ = dispatch(ctx, "bash", {"cmd": "ls", "timeout_s": 60})
    assert "ceiling" not in content


def test_a_timed_out_command_explains_its_minus_one(ctx, backend):
    """A bare `exit_code: -1` reads like a command that merely produced nothing.
    Found by a live run, where the model had to re-issue the query to find out."""
    backend.exec_result = ExecResult(-1, "", False, None)
    content, _ = dispatch(ctx, "bash", {"cmd": "sleep 280", "timeout_s": 120})
    assert "exit_code -1 means no exit status was reported" in content
    assert "ran with 120s" in content
    assert "job_start has no time limit" in content


def test_an_ordinary_exit_says_nothing_about_timeouts(ctx, backend):
    backend.exec_result = ExecResult(0, "fine", False, None)
    content, _ = dispatch(ctx, "bash", {"cmd": "ls"})
    assert "exit_code -1" not in content


# -- a command the workspace moved to a job -----------------------------------
#
# Measured in production on 2026-09-21 (study 20260921-033019-e1b4, workspace
# 35c9f018): `bash sleep 240`, `sleep 200` and `sleep 180`, each asked with
# `timeout_s=300`, came back after the service's 120 s synchronous window as
# `exit_code: 0` -- the hosted backend turned the service's `{exit_code: null,
# promoted: true, job_id}` into `ExecResult(exit_code=0, output=note, job_id=...)`, and
# `_bash` printed the zero first. The model read three commands that had finished with
# no output, polled the first job once (two seconds in) and never again, and the job
# rows (808edf10, 5432459d, e839ac57) read `running` for 46, 35 and 20 minutes after
# their commands had ended. The caller asked for 300 s; it gets them.


class _Clock:
    """A clock the tests turn by hand, so a 300 s wait costs no seconds.

    `_bash` measures the exec with `time.monotonic` and `_hold` sleeps between polls;
    both are read off `openreynolds.tools.time`, so replacing them there turns every
    `sleep` into a step of the same clock."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr("openreynolds.tools.time.monotonic", clock.monotonic)
    monkeypatch.setattr("openreynolds.tools.time.sleep", clock.sleep)
    return clock


PROMOTED_JOB = "808edf10-0957-482f-a105-2151008d3a86"
"""The first of the three, as the transcript names it."""


def _promotes(backend, clock, *, after=120.0, output="", job_id=PROMOTED_JOB):
    """The hosted backend's answer to a command still running at the window: the exec
    call comes back after `after` seconds with no exit code, the flag and the job,
    which is running with an empty log until a test says otherwise."""

    def exec(cmd, cwd=None, timeout_s=120, *, background=False):
        backend.last_exec = (cmd, cwd, timeout_s)
        backend.execs.append(cmd)
        clock.sleep(after)
        return ExecResult(None, output, False, None, job_id=job_id, promoted=True)

    backend.exec = exec
    backend.jobs[job_id] = JobStatus(job_id=job_id, status="running", log_size=0)
    backend.logs[job_id] = b""


def _ends_after(backend, polls, *, exit_code=0, log=b"", job_id=PROMOTED_JOB):
    """The job ends on the `polls`-th status read, with this exit code and log."""
    reads = {"n": 0}
    real = backend.job_status

    def status(jid):
        reads["n"] += 1
        if jid == job_id and reads["n"] >= polls:
            backend.jobs[jid] = JobStatus(
                job_id=jid, status="exited", exit_code=exit_code,
                end_reason="completed" if exit_code == 0 else "failed", log_size=len(log),
            )
            backend.logs[jid] = log
        return real(jid)

    backend.job_status = status
    return reads


def test_a_promoted_command_is_waited_out_and_answers_with_the_jobs_result(ctx, backend, store, clock):
    """The caller asked for 300 s and the service held the command for 120 of them
    before moving it to a job: the tool waits the rest on the job, and hands back the
    job's exit code and log as the command's own -- the promotion made transparent."""
    _promotes(backend, clock)
    _ends_after(backend, 3, exit_code=0, log=b"Time = 0.5\nEnd\n")

    content, is_error = dispatch(ctx, "bash", {"cmd": "sleep 240; ls -R mesh | head -40",
                                               "timeout_s": 300})

    assert not is_error
    assert content.startswith("exit_code: 0\n"), content
    assert "still running as job" not in content
    assert PROMOTED_JOB in content and "moved to job" in content, "it names the job it followed"
    assert "Time = 0.5\nEnd" in content, "the job's log is the command's output"
    assert "[took 130s]" in content, "120 s in the exec, two polls of 5 s waiting"
    assert backend.last_exec == ("sleep 240; ls -R mesh | head -40", WORKSPACE_ROOT, 300)
    record = store.session.jobs[PROMOTED_JOB]
    assert record.status == "exited" and record.exit_code == 0
    assert record.cmd == "sleep 240; ls -R mesh | head -40"
    assert record.log_offset == len(b"Time = 0.5\nEnd\n"), "a job_check next continues, not repeats"


def test_a_promoted_command_that_fails_in_its_job_answers_with_that_exit_code(ctx, backend, clock):
    _promotes(backend, clock)
    _ends_after(backend, 2, exit_code=1, log=b"--> FOAM FATAL ERROR\n")

    content, is_error = dispatch(ctx, "bash", {"cmd": "reconstructPar", "timeout_s": 300})

    assert not is_error
    assert content.startswith("exit_code: 1\n"), content
    assert "FOAM FATAL" in content
    assert "exit_code: 0" not in content


def test_a_promoted_command_still_running_at_the_callers_timeout_leads_with_that_fact(ctx, backend, store, clock):
    """`sleep 240` with `timeout_s=300`: 120 s in the exec, then the job is followed for
    the 180 s left -- and the job, running the command again from the start, has not
    ended. The first line is the fact, never `exit_code: 0`; the job is left in the
    session's record as running, so the watch loop polls it and wakes the model when
    it ends, which is the poll that never happened in production."""
    _promotes(backend, clock)
    backend.logs[PROMOTED_JOB] = b"still sleeping\n"

    content, is_error = dispatch(ctx, "bash", {"cmd": "sleep 240", "timeout_s": 300})

    assert not is_error
    first, rest = content.split("\n", 1)
    assert first == f"still running as job {PROMOTED_JOB} (promoted after 120 s; 180 s waited)"
    assert "exit_code: 0" not in content
    assert f"job_check {PROMOTED_JOB}" in rest, "what follows it from here"
    assert "not a command to run again" in rest
    assert "still sleeping" in rest, "the log so far"
    assert clock.now == 300.0, "the caller's own timeout, and not a second longer"
    assert store.session.jobs[PROMOTED_JOB].status == "running"
    assert [job.job_id for job in store.live_jobs()] == [PROMOTED_JOB], "left for the watch loop"


def test_a_promoted_wait_ends_early_when_the_person_writes(ctx, backend, clock):
    """The same interruption every held wait honours (`ToolContext.on_wait_input`), with
    one difference in the note: the next call is `job_check`, because `bash` again
    would run the command a third time."""
    _promotes(backend, clock)
    ctx.on_wait_input = lambda: True

    content, is_error = dispatch(ctx, "bash", {"cmd": "sleep 240", "timeout_s": 300})

    assert not is_error
    assert content.startswith(f"still running as job {PROMOTED_JOB} (promoted after 120 s; 0 s waited)")
    assert "the person wrote, so this answered early" in content
    assert "their words follow this result" in content
    assert f"follow it with job_check {PROMOTED_JOB}" in content
    assert "call bash again" not in content
    assert clock.now == 120.0


def test_a_promoted_wait_ends_when_the_person_leaves(ctx, backend, clock):
    """`ToolContext.on_leaving`, wired to `Loop.leaving`: End pressed during the wait
    ends it within a poll, with the note the other waits use."""
    _promotes(backend, clock)
    ctx.on_leaving = lambda: True

    content, is_error = dispatch(ctx, "bash", {"cmd": "sleep 240", "timeout_s": 300})

    assert not is_error
    assert content.startswith(f"still running as job {PROMOTED_JOB} (promoted after 120 s; 0 s waited)")
    assert "the person ended the session, so this answered early" in content
    assert "exit_code" not in content.split("\n")[0]


def test_a_promotion_with_no_time_left_is_still_reported_as_the_job_it_is(ctx, backend, clock):
    """A `timeout_s` barely above the window, spent by the exec itself: nothing to wait
    with, and the answer is still the job, not a zero."""
    _promotes(backend, clock, after=125.0)

    content, _ = dispatch(ctx, "bash", {"cmd": "sleep 200", "timeout_s": 121})

    assert content.startswith(f"still running as job {PROMOTED_JOB} (promoted after 125 s; 0 s waited)")
    assert clock.now == 125.0, "no wait was made with a budget already spent"


def test_a_promoted_command_whose_job_cannot_be_read_keeps_the_promotion_in_front(ctx, backend, clock):
    """The transport can fail between the promotion and the answer. The promotion is
    the fact that must not be lost behind that: the first line stands and the
    failure is a note."""
    from openreynolds.backend.base import BackendError

    _promotes(backend, clock)

    def gone(job_id):
        raise BackendError("the workspace did not answer", code="timeout")

    backend.job_status = gone
    content, is_error = dispatch(ctx, "bash", {"cmd": "sleep 240", "timeout_s": 300})

    assert not is_error
    assert content.startswith(f"still running as job {PROMOTED_JOB}")
    assert "could not be read just now" in content and "did not answer" in content


def test_the_synchronous_runs_output_is_kept_apart_from_the_jobs_log(ctx, backend, clock):
    """Two runs of one command: what the first printed before it was moved, then the
    job's log from the start. Labelled, so a repeated opening line reads as what it is."""
    _promotes(backend, clock, output="starting\n")
    _ends_after(backend, 2, log=b"starting\ndone\n")

    content, _ = dispatch(ctx, "bash", {"cmd": "echo starting; sleep 150; echo done",
                                        "timeout_s": 300})

    assert "first run, before it was moved" in content
    assert content.index("first run") < content.index("the job's log, from the start")
    assert content.endswith("starting\ndone\n")


def test_a_timeout_within_the_window_is_formatted_exactly_as_before(ctx, backend, clock):
    """Nothing about an ordinary command changed: a `timeout_s` the service never
    promotes gives the same bytes it always did."""
    backend.exec_result = ExecResult(0, "ok", False, None)
    for timeout_s in (60, 120):
        content, is_error = dispatch(ctx, "bash", {"cmd": "ls", "timeout_s": timeout_s})
        assert not is_error
        assert content == "exit_code: 0\n\nok", content
    assert "job" not in content


def test_the_bash_description_says_a_long_command_becomes_a_job():
    """The model reads the description, not this file. It has to learn there that a
    command still running at the window comes back as a job it can `job_check` -- and
    learn it as a fact about the environment, not as an instruction."""
    import re

    from openreynolds.backend.base import EXEC_SYNC_WINDOW_S
    from test_prompt import IMPERATIVE_PATTERNS

    description = next(t for t in TOOLS if t["name"] == "bash")["description"]
    assert f"{EXEC_SYNC_WINDOW_S} seconds" in description
    assert "still running as job" in description and "job_check" in description
    for pattern in IMPERATIVE_PATTERNS:
        assert not re.search(pattern, description, re.IGNORECASE), pattern


# -- a study works in its own directory ----------------------------------------


def test_a_command_runs_in_the_study_s_own_directory(ctx):
    """Otherwise every study's relative paths land in the same shared heap."""
    ctx.home = "/work/20260824-120000-abcd"

    dispatch(ctx, "bash", {"cmd": "ls"})

    assert ctx.backend.last_exec[1] == "/work/20260824-120000-abcd"


def test_an_explicit_directory_still_wins(ctx):
    ctx.home = "/work/mine"
    dispatch(ctx, "bash", {"cmd": "ls", "cwd": "/work/somewhere-else"})
    assert ctx.backend.last_exec[1] == "/work/somewhere-else"


def test_a_job_starts_in_the_study_s_own_directory_too(ctx):
    ctx.home = "/work/mine"
    dispatch(ctx, "job_start", {"cmd": "simpleFoam", "name": "solve"})
    assert ctx.backend.started[-1]["cwd"] == "/work/mine"


def test_the_default_is_the_whole_workspace_when_no_home_is_set(ctx):
    """Studies made before studies had a directory of their own keep what they had."""
    dispatch(ctx, "bash", {"cmd": "ls"})
    assert ctx.backend.last_exec[1] == WORKSPACE_ROOT


# -- how long things took ------------------------------------------------------


def test_a_slow_command_says_how_long_it_took(ctx, monkeypatch):
    """A four-minute command and a two-second one read identically otherwise, so the
    cost of what was just done is invisible to whoever chose to do it."""
    clock = iter([0.0, 42.0])
    monkeypatch.setattr("openreynolds.tools.time.monotonic", lambda: next(clock))

    out, _ = dispatch(ctx, "bash", {"cmd": "blockMesh"})

    assert "[took 42s]" in out


def test_a_quick_command_is_not_cluttered_with_a_duration(ctx, monkeypatch):
    clock = iter([0.0, 0.3])
    monkeypatch.setattr("openreynolds.tools.time.monotonic", lambda: next(clock))

    out, _ = dispatch(ctx, "bash", {"cmd": "ls"})

    assert "took" not in out


def test_a_running_job_says_how_long_it_has_been_running(ctx):
    """Two hours in and one minute in are the same line otherwise, and which of those
    it is changes what anyone would do about it."""
    ctx.backend.jobs["job-1"] = JobStatus(
        job_id="job-1",
        status="running",
        name="solve",
        started_at=time.time() - 3600,
    )
    ctx.store.record_job("job-1", cmd="simpleFoam", name="solve")

    out, _ = dispatch(ctx, "job_check", {"job_id": "job-1"})

    assert "running_for=60." in out


def test_a_finished_job_says_how_long_it_ran(ctx):
    ctx.backend.jobs["job-1"] = JobStatus(
        job_id="job-1",
        status="exited",
        name="solve",
        exit_code=0,
        started_at="2026-08-24T10:00:00Z",
        ended_at="2026-08-24T10:03:00Z",
    )
    ctx.store.record_job("job-1", cmd="simpleFoam", name="solve")

    out, _ = dispatch(ctx, "job_check", {"job_id": "job-1"})

    assert "ran_for=3.0min" in out


def test_a_job_with_no_timestamps_says_nothing_about_duration(ctx):
    """The service does not always send them, and a made-up number is worse than none."""
    ctx.backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")
    ctx.store.record_job("job-1", cmd="simpleFoam", name="solve")

    out, _ = dispatch(ctx, "job_check", {"job_id": "job-1"})

    assert "running_for" not in out and "ran_for" not in out


# -- job_check can wait ---------------------------------------------------------


def test_job_check_waits_for_the_job_to_end(ctx, monkeypatch):
    """The model paced itself with `sleep` in bash, which tripped the bash cap and
    filled the transcript with timeout noise. Waiting is the harness's job."""
    import time as _time

    from openreynolds import tools as tools_mod
    from openreynolds.backend.base import JobStatus
    from openreynolds.tools import dispatch as _dispatch

    monkeypatch.setattr(tools_mod, "JOB_WAIT_POLL_S", 0.01)
    job_id = ctx.backend.job_start("simpleFoam", name="solve")
    ctx.store.record_job(job_id, cmd="simpleFoam", name="solve")
    calls = {"n": 0}
    real = ctx.backend.job_status

    def finishing(jid):
        calls["n"] += 1
        if calls["n"] >= 3:
            ctx.backend.jobs[jid] = JobStatus(
                job_id=jid, name="solve", status="exited", exit_code=0,
                end_reason="completed", log_size=0,
            )
        return real(jid)

    ctx.backend.job_status = finishing

    out, is_error = _dispatch(ctx, "job_check", {"job_id": job_id, "wait_s": 5})

    assert not is_error
    assert "exited" in out
    assert "waited" in out


def test_the_wait_ends_early_when_the_user_speaks(ctx, monkeypatch):
    import time as _time

    from openreynolds import tools as tools_mod
    from openreynolds.tools import dispatch as _dispatch

    monkeypatch.setattr(tools_mod, "JOB_WAIT_POLL_S", 0.01)
    job_id = ctx.backend.job_start("simpleFoam", name="solve")
    ctx.store.record_job(job_id, cmd="simpleFoam", name="solve")
    ctx.on_wait_input = lambda: True

    began = _time.monotonic()
    out, is_error = _dispatch(ctx, "job_check", {"job_id": job_id, "wait_s": 30})

    assert not is_error
    assert _time.monotonic() - began < 5, "it did not sit out the full wait"
    assert "the person wrote, so this answered early" in out
    assert "their words follow this result" in out
    assert "call job_check again -- the job is still running" in out


def test_a_wait_cut_by_the_person_says_so_although_asking_again_finds_nothing(ctx, monkeypatch):
    """The question drains the inbox to answer (`Loop.heard`), so asked a second time
    after the loop it says no. The note used to be decided by that second asking, which
    would have dropped the one sentence that explains a wait of 0 s."""
    from openreynolds import tools as tools_mod
    from openreynolds.tools import dispatch as _dispatch

    monkeypatch.setattr(tools_mod, "JOB_WAIT_POLL_S", 0.01)
    job_id = ctx.backend.job_start("simpleFoam", name="solve")
    ctx.store.record_job(job_id, cmd="simpleFoam", name="solve")
    answers = iter([True])
    asked = []
    ctx.on_wait_input = lambda: asked.append(1) or next(answers, False)

    out, is_error = _dispatch(ctx, "job_check", {"job_id": job_id, "wait_s": 30})

    assert not is_error
    assert len(asked) == 1, "asked once in the loop, and not again after it"
    assert "the person wrote, so this answered early" in out


def test_an_over_long_wait_is_clamped_and_says_so(ctx, monkeypatch):
    from openreynolds import tools as tools_mod
    from openreynolds.tools import dispatch as _dispatch

    monkeypatch.setattr(tools_mod, "JOB_WAIT_POLL_S", 0.01)
    monkeypatch.setattr(tools_mod, "JOB_WAIT_MAX_S", 0.05)
    job_id = ctx.backend.job_start("simpleFoam", name="solve")
    ctx.store.record_job(job_id, cmd="simpleFoam", name="solve")

    out, is_error = _dispatch(ctx, "job_check", {"job_id": job_id, "wait_s": 9999})

    assert not is_error
    assert "ceiling" in out


def test_no_wait_asked_means_no_waiting(ctx):
    import time as _time

    from openreynolds.tools import dispatch as _dispatch

    job_id = ctx.backend.job_start("simpleFoam", name="solve")
    ctx.store.record_job(job_id, cmd="simpleFoam", name="solve")

    began = _time.monotonic()
    out, is_error = _dispatch(ctx, "job_check", {"job_id": job_id})

    assert not is_error
    assert _time.monotonic() - began < 1
    assert "waited" not in out


# -- the same bytes are not sent twice -----------------------------------------


@pytest.fixture
def roomy(backend, store):
    """A context whose output cap is above the echo threshold, so a large repeat is
    actually large by the time it is compared."""
    return ToolContext(backend=backend, store=store, max_output=50_000)


def test_a_byte_identical_repeat_points_at_the_first_one(roomy):
    """`bash` output is 78-82% of everything the model is sent, and one study's most
    expensive call -- `cat .toolbox/notes/*.md` -- was 4,177 tokens that then rode along
    in all 76 requests after it. Reading it again puts a second copy in the same thread."""
    roomy.backend.exec_result = ExecResult(0, "N" * 5_000, False, None)

    first, _ = dispatch(roomy, "bash", {"cmd": "cat notes.md"})
    assert "N" * 5_000 in first

    again, error = dispatch(roomy, "bash", {"cmd": "cat notes.md"})
    assert not error, "the call succeeded; only the second copy of the bytes is gone"
    assert "identical, byte for byte" in again
    assert "#1" in again, "and it says where the bytes already are"
    assert len(again) < 500


def test_output_that_changed_is_never_collapsed(roomy):
    """A command whose answer moved is exactly the interesting case."""
    roomy.backend.exec_result = ExecResult(0, "A" * 5_000, False, None)
    dispatch(roomy, "bash", {"cmd": "tail log"})
    roomy.backend.exec_result = ExecResult(0, "B" * 5_000, False, None)
    second, _ = dispatch(roomy, "bash", {"cmd": "tail log"})
    assert "B" * 5_000 in second


def test_a_short_repeat_is_left_exactly_as_it_is(ctx):
    """Below the threshold the sentence explaining the repeat costs what the repeat does."""
    ctx.backend.exec_result = ExecResult(0, "ok", False, None)
    dispatch(ctx, "bash", {"cmd": "ls"})
    again, _ = dispatch(ctx, "bash", {"cmd": "ls"})
    assert "identical" not in again and "ok" in again


def test_the_same_failure_twice_is_still_reported_twice(ctx):
    """A repeated error is a fact about the run, not a duplicate to fold away."""
    def refuse(cmd, cwd=None, timeout_s=120):
        raise BackendError("x" * 5_000, code="boom")

    ctx.backend.exec = refuse
    first, error1 = dispatch(ctx, "bash", {"cmd": "go"})
    second, error2 = dispatch(ctx, "bash", {"cmd": "go"})
    assert error1 and error2
    assert first == second and "identical" not in second
