"""The mesh desk in the background: a run on its own thread, watched rather than waited on.

Why this exists. The `mesh` tool used to call `Mesher.run` on the loop's own thread and
hold it for the whole build. Measured in production (study 20260920-161908-c7ef): one
call held the agent for 402 s and some twenty-five model calls. For all of that time
the main agent could answer nothing; the lines the person typed were drained INTO the
desk by its `interject`, which was the same inbox the loop reads between its own tool
calls; and on the web page each typed line sat marked as pending until the desk
consumed it, which the person experienced as "sending a message takes minutes". The
desk's work was fine. Where it ran was not.

So a run lives here, on a daemon thread, and the loop's thread is free: the tool call
returns at once, the main agent goes on talking, the desk's steps are shown as they
happen, and the finish wakes the agent through the same watch loop a job's end does
(`watch.watch`). The desk gets ears of its own -- `note` and `take_remarks` -- so what
the person types reaches it only when the main agent decides it should (`mesh_note`),
and the session's inbox is the main agent's again.

Nothing here changes how the desk meshes: `Mesher.run` is called exactly as before,
with different callbacks. One run at a time per session; a second `mesh` call while
one runs is answered with where the first has got to.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable

from ..backend.base import WORKSPACE_ROOT
from .agent import MeshResult, Step, _case_name


class DeskRun:
    """One background build: the desk's thread, what it has done so far, and its answer.

    `mesher` is the session's desk (`Mesher`, or anything with its shape). The desk
    reads its `interject` and `on_step` off the instance it runs on, and the session's
    instance listens to the session's inbox -- which is precisely what a background run
    must not do -- so the run is made on a shallow copy of it with this run's ears and
    this run's step record in place. The copy shares the workspace, the store, the
    configuration and the provider client, all of which are meant to be shared; the
    two things it does not share are the two that were the problem.
    """

    def __init__(self, mesher: Any, request: str, case: str | None = None,
                 on_step: Callable[[Step], None] | None = None):
        self.request = request
        self.case = case
        self.case_rel = _case_name(case)
        self.started_at = time.monotonic()
        """Monotonic, for elapsed time."""
        self.started_wall = _now()
        """Wall clock, for the session record a resume reads."""
        self.steps: list[Step] = []
        self.result: MeshResult | None = None
        self.error = ""
        """Set when the run raised rather than returned -- a harness fault, since
        `Mesher.run` reports the desk's own failures inside its result."""
        self.done = threading.Event()
        self.delivered = False
        """Whether the result has been handed to the calling agent (tokens counted,
        the registry cleared). Guarded so a wake and a `mesh_wait` racing for the same
        finished run cannot count its tokens twice."""
        self._remarks: list[str] = []
        self._lock = threading.Lock()
        self._show = on_step if on_step is not None else getattr(mesher, "on_step", None)
        self.mesher = copy.copy(mesher)
        self.mesher.interject = self.take_remarks
        self.mesher.on_step = self._step
        self._thread = threading.Thread(
            target=self._work, name=f"mesh-desk-{self.case_rel}", daemon=True,
        )

    # -- the thread ------------------------------------------------------------

    def start(self) -> "DeskRun":
        self._thread.start()
        return self

    def _work(self) -> None:
        try:
            self.result = self.mesher.run(self.request, case=self.case)
        except Exception as exc:  # noqa: BLE001 - a thread's exception is a silent one
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.done.set()

    def abandon(self) -> None:
        """End the run at its next lap: the session that started it is ending.

        The desk's loop reads its budgets at every lap, so zeroing them on this run's
        copy ends the run after the command in flight, with the finish check still run
        on whatever is on disk; nothing in the desk's own loop changes for this. The
        thread is a daemon and dies with a process that exits, so the terminal needs
        none of it. A process that hosts several sessions and does not exit -- the
        hosted runner -- does: a desk left building for a session that has ended would
        spend its whole budget, model calls included, on nobody.
        """
        try:
            self.mesher.max_steps = 0
            self.mesher.max_seconds = 0.0
        except Exception:  # noqa: BLE001 - a desk without budgets has nothing to stop
            pass

    def _step(self, step: Step) -> None:
        """Record the step, then show it the way a foreground step was shown."""
        with self._lock:
            self.steps.append(step)
        if self._show is not None:
            try:
                self._show(step)
            except Exception:  # noqa: BLE001 - a progress line may never end a mesh
                pass

    # -- the person's words ----------------------------------------------------

    def note(self, text: str) -> None:
        """Queue a remark for the desk; it reads it at its next step."""
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._remarks.append(text)

    def take_remarks(self) -> str | None:
        """What `Mesher._remark` reads: everything noted since the last step, or None."""
        with self._lock:
            taken, self._remarks = self._remarks, []
        return "\n".join(taken) or None

    # -- what to say about it --------------------------------------------------

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def record(self) -> dict[str, str]:
        """What the session remembers about a live run, for a resume that finds it."""
        return {"case_rel": self.case_rel, "request": self.request,
                "started_at": self.started_wall}

    def progress_line(self) -> str:
        """One line of fact: the case, how long, how many steps, the last command."""
        with self._lock:
            steps = list(self.steps)
        minutes, seconds = divmod(int(self.elapsed_s), 60)
        count = f"{len(steps)} step{'' if len(steps) == 1 else 's'}"
        state = "finished after" if self.done.is_set() else "running for"
        line = f"mesh desk on `{self.case_rel}`: {state} {minutes}m{seconds:02d}s, {count}"
        if steps:
            last = steps[-1]
            first = (last.cmd or "").strip().splitlines()
            head = first[0][:80] if first else ""
            line += f"; last command `{head}` exited {last.exit_code}"
            line += ", with a picture" if last.image else ", no picture"
        return line

    def picture(self) -> str:
        """Where the desk's own picture of the mesh is on the workspace, or ''."""
        result = self.result
        if result is None or not result.png or result.check is None:
            return ""
        return result.check.render_abs or result.check.render or ""

    def report(self) -> str:
        """The finished run as a wake message: the same words the tool would have
        returned, and where the picture is. The picture itself is not attached -- a
        wake is text, and `read_file` on the path shows it when the agent wants it."""
        if self.result is None:
            return (f"The mesh desk on `{self.case_rel}` stopped without a result "
                    f"({self.error or 'no reason recorded'}). Whatever it built is on "
                    f"disk under `{self.case_rel}`.")
        lines = [f"The mesh desk on `{self.case_rel}` has finished.", mesh_text(self.result)]
        where = self.picture()
        if where:
            lines.append(f"the desk's picture of the mesh is at {where}; read_file shows it.")
        return "\n".join(lines)


def mesh_text(result: Any) -> str:
    """The words of the mesh tool's answer: whether it is a mesh, what the mesh is,
    what the desk says it built, what is still to do, and how to change it.

    The order is deliberate. A tool result that opened with "case written" was once
    read as "meshed" and the solve that followed had nothing to solve, so the first
    line here is always the state of `constant/polyMesh` and never anything else.

    Here rather than in `tools.py` because both ends of a run read it -- the tool that
    returns a foreground result and the wake that reports a background one -- and
    `tools.py` imports this module, so this module cannot import it back.
    """
    check = result.check
    lines: list[str] = []
    if result.error and not result.ok:
        lines.append(f"the mesh desk stopped: {result.error}")
    elif result.error:
        lines.append(f"the mesh desk stopped ({result.error}) -- but the mesh it had "
                     "already built is there and passes:")
    if result.ok and check is not None:
        lines.append(f"meshed: {result.case_rel}/constant/polyMesh is an OpenFOAM mesh "
                     "and checkMesh passes on it.")
    elif check is not None and check.unreachable:
        # Nothing is known: the workspace did not answer. Three runs had a finished
        # mesh described as missing because a container recycled while it was being
        # checked, and the caller believed it.
        lines.append(
            f"the mesh in {result.case_rel} could NOT BE CHECKED -- the workspace did "
            f"not answer ({result.check.error}). This is not a statement about the "
            "mesh: a container that recycles mid-run comes back and the Volume under "
            f"it keeps the files, so look for yourself with `python3 "
            f"{WORKSPACE_ROOT}/.toolbox/mesh_look.py {result.case_rel} --out look.png` "
            "before building anything again.")
    elif check is not None and check.missing:
        why = "; ".join(check.missing)
        lines.append(f"NOT a usable mesh yet in {result.case_rel}: {why}")
    else:
        lines.append(f"nothing was meshed in {result.case_rel}")
    if getattr(result, "remarks", None):
        lines.append("")
        lines.append("while this ran, the user said this to the mesh desk directly, and it "
                     "worked to it:")
        lines.extend(f'  "{remark}"' for remark in result.remarks)
    if result.summary:
        lines.append("")
        lines.append("the mesh desk says:")
        lines.extend(f"  {line}" for line in result.summary.splitlines())
    if check is not None and check.lines():
        lines.append("")
        lines.extend(check.lines())
    lines.append("")
    lines.append(_mesh_accounting(result))
    lines.append(
        f"this is a mesh and nothing else: no 0/ fields, no boundary conditions, no "
        f"solver settings, nothing solved. Look at it again yourself with "
        f"`python3 {WORKSPACE_ROOT}/.toolbox/mesh_look.py {result.case_rel} --out look.png`, "
        f"rebuild it after an edit with `cd {result.case_rel} && sh Allmesh`, or call this "
        "tool again with what to change."
    )
    return "\n".join(lines)


def _mesh_accounting(result: Any) -> str:
    steps = len(getattr(result, "steps", []) or [])
    line = f"{steps} step{'s' if steps != 1 else ''}, {result.seconds / 60:.1f} min"
    if result.stopped == "steps":
        line += f"; it ran out of steps before it was finished, so this is where it got to"
    elif result.stopped == "time":
        line += "; it ran out of time before it was finished, so this is where it got to"
    elif result.stopped == "provider":
        line += "; the model call failed, so this is where it got to"
    return line


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
