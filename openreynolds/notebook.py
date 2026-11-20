"""Jupyter: the two things a notebook wants that a script does not.

The library surface is already here. `openreynolds.start`, `Study.events`, `Study.wait`,
`Study.results` and `Study.file` are the nine capabilities of `mcp/core.py` as Python
objects (`sdk.py:1-23`), and a notebook is Python, so driving a study from a cell needs
nothing new:

    import openreynolds
    study = openreynolds.start("drag on a sphere at Re=1000, SI units")
    openreynolds.follow(study)      # one cell, rewritten as it goes
    openreynolds.show(study)        # the answer, with its figures in the output

What a notebook does not get from the SDK is those two lines, and they are the whole of
this module:

**`follow`** -- `for event in study.events(): print(...)` works in a terminal and is
wrong in a notebook: each status answer appends, and a forty-minute study leaves a cell
with hundreds of scrolled lines. A notebook's equivalent of a moving line is
`clear_output(wait=True)` and one re-render, which is what this does.

**`show`** -- `results()` reports its pictures as `{"name", "url"}` where the url is
this machine's path to the file (`mcp/local.py:698-699`), under the renders directory
the mirror has already filled. A notebook can display those; a dict of paths makes the
reader open them by hand. The figures are the result of a CFD study, so they belong in
the output of the cell that asked for it.

Neither function decides anything about the study and neither is required to use it:
`openreynolds.show(study)` returns the same dict `study.results()` does, so a notebook
that would rather render it itself loses nothing by never calling this.

IPython is **not a dependency of this package** and must not become one -- a CLI that
pulls in a notebook stack to print a study id is the wrong trade. Both functions work
without it: with no IPython on the path they fall back to printing, which is what they
would have done anyway.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PICTURE_SUFFIXES = (".png", ".gif", ".jpg", ".jpeg", ".svg", ".webp")
"""What `show` will put in the output. `.mp4`/`.webm` are deliberately left out: a
video is shown with `IPython.display.Video`, which needs the file served or embedded
whole, and a notebook is not where this repo's videos are watched (`openreynolds video`
and `renders` are)."""

MAX_PICTURES = 12
"""How many figures go in one cell, newest last.

`core.study_results` already hands back only the last 20 renders
(`mcp/core.py:751-754`). Twelve of those is a screen or two of scrolling; the rest are
named in the dict the call returns, which is the complete list."""


def _display():
    """IPython's `display` module, or `None` outside a notebook.

    Imported here rather than at the top so that importing `openreynolds` stays the
    cheap import `__init__.py` is written to keep it."""
    try:
        from IPython import display
    except ImportError:
        return None
    return display


def _pictures(results: dict[str, Any]) -> list[Path]:
    """The figures that are on this machine, in the order they were made.

    A picture whose file is not here is left out rather than shown as a broken image:
    the mirror reports every file it did not bring and why (`mirror.py:30-37`), and a
    notebook cell is not where that argument belongs. The dict `show` returns still
    lists it.
    """
    found = []
    for row in results.get("pictures") or []:
        url = str(row.get("url") or "")
        if not url or not url.lower().endswith(PICTURE_SUFFIXES):
            continue
        path = Path(url)
        if path.is_file():
            found.append(path)
    return found[-MAX_PICTURES:]


def _summary(results: dict[str, Any]) -> str:
    """The study, its jobs and its answer, as markdown."""
    lines = [f"**{results.get('title') or results.get('study_id') or 'study'}**",
             f"`{results.get('study_id') or ''}`  --  `{results.get('workspace') or ''}`"]
    jobs = results.get("jobs") or []
    if jobs:
        lines.append("")
        lines.append("| job | status | exit |")
        lines.append("| --- | --- | --- |")
        for job in jobs[-10:]:
            lines.append(f"| {job.get('name') or ''} | {job.get('status') or ''} "
                         f"| {job.get('exit_code') if job.get('exit_code') is not None else ''} |")
    answer = results.get("final_message")
    if answer:
        lines += ["", str(answer)]
    return "\n".join(lines)


def show(study: Any, *, pictures: bool = True) -> dict[str, Any]:
    """Put a study's answer and its figures in this cell's output, and return the dict.

    `study` is a `Study` from `start()` or a study id. The return value is exactly
    `results()`'s, which lists every picture and every file including the ones not
    shown here.
    """
    from . import sdk

    results = study.results() if hasattr(study, "results") else sdk.results(str(study))
    display = _display()
    if display is None:
        print(_summary(results))
        for path in _pictures(results) if pictures else []:
            print(path)
        return results
    display.display(display.Markdown(_summary(results)))
    for path in _pictures(results) if pictures else []:
        display.display(display.Image(filename=str(path)))
    return results


def follow(study: Any, *, wait_s: float | None = None) -> dict[str, Any]:
    """Watch a running study in one cell, rewritten as it goes, and return its last state.

    Blocks until the study hands the turn back -- waiting for input, ended, or broken,
    which is `sdk.DONE` -- the same point `Study.events` stops at. Interrupting the cell
    leaves the study running; it is `study.end()` (or `openreynolds.shutdown()`) that
    stops paying for the workspace, and nothing here does that on your behalf, because
    a cell interrupted while reading is not a decision to throw the work away.
    """
    display = _display()
    answer: dict[str, Any] = {}
    seen: list[dict[str, Any]] = []
    kwargs = {} if wait_s is None else {"wait_s": wait_s}
    for answer in study.events(**kwargs):
        # Each answer carries only the rows after the cursor `Study.status` moved on
        # (`sdk.py:78-80,100`), so the history has to be kept here. Re-rendering only
        # the newest batch would make a cell that forgets the study as it runs.
        seen += [row for row in (answer.get("transcript") or []) if isinstance(row, dict)]
        text = _progress(answer, seen)
        if display is None:
            print(text, flush=True)
            continue
        display.clear_output(wait=True)
        display.display(display.Markdown(text))
    return answer


TAIL_ROWS = 10
"""How many transcript rows the followed cell keeps on screen. It is rewritten in
place, so this is a window, not a budget: the whole transcript is in `study.results()`.
"""


def _row(row: dict[str, Any]) -> str:
    """One condensed transcript row as a line. `core.condense` names the speaker four
    different ways depending on what the row is -- `from` for a message, `tool` for a
    call, `desk` for a desk step, `event` for everything the session did
    (`mcp/core.py:421-445`) -- and all four have to read as a line here."""
    if row.get("tool"):
        said = str(row.get("input") or "") or str(row.get("output") or "")
        mark = " (error)" if row.get("error") else ""
        return f"- `{row['tool']}`{mark} {said[:200]}".rstrip()
    if row.get("desk"):
        return f"- desk {row['desk']}: {str(row.get('text') or '')[:200]}".rstrip()
    if row.get("event"):
        return f"- {str(row['event'])[:200]}"
    who = str(row.get("from") or "")
    return f"- **{who}**: {str(row.get('text') or '')[:300]}".rstrip()


def _progress(answer: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """One cell's worth of "where is it now", as markdown.

    The rows are the condensed transcript `core.status_answer` already built, so this
    renders what the MCP tool and the HTTP route report and never a second reading of
    the session's state.
    """
    head = f"**{answer.get('state') or '?'}**"
    for key in ("stage", "narration"):
        if answer.get(key):
            head += f" -- {answer[key]}"
            break
    if answer.get("study_id"):
        head += f"  `{answer['study_id']}`"
    lines = [head, ""] + [_row(row) for row in rows[-TAIL_ROWS:]]
    for job in (answer.get("jobs") or [])[-5:]:
        lines.append(f"- job `{job.get('name')}` {job.get('status')}")
    for approval in answer.get("approvals") or []:
        lines += ["", f"**waiting for you:** {approval.get('title') or ''} "
                      f"({', '.join(approval.get('choices') or [])})"]
    if answer.get("error"):
        lines += ["", f"**error:** {answer['error']}"]
    return "\n".join(lines)
