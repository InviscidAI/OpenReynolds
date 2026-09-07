"""The loop: one bash block a step, on the machine with OpenFOAM, until the mesh checks out.

Small on purpose. There is no tool schema, no state machine, no spec language and no
compiler: a thread of messages, a fenced command taken out of each one, the command's
output put back in, and a picture alongside it whenever the command drew one. Every
mechanism this file does not have was in the thing it replaces, and none of them were
what made a mesh right.

Three things it does own, because they are what the model cannot check about itself:

* **the picture arrives** -- a command that writes a PNG has that PNG attached to its
  output, so seeing is not something the agent has to remember to arrange;
* **the finish is verified** -- `echo MESH_DONE` does not end the run, `check.py` does,
  and a failed check is handed back as work rather than reported as success;
* **the budget** -- steps and wall clock, so a run that is going nowhere stops going
  there.
"""

from __future__ import annotations

import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import images
from ..llm import Listener, ProviderError, make_provider
from .brief import MESH_DONE, remark_message, system_prompt, task_message
from .check import Check, verify

MAX_STEPS = 30
"""Enough for a shape, a look, two or three revisions, a refine and a finish. A run
that has not converged by here is not one step from converging."""

MAX_SECONDS = 1200.0
"""Twenty minutes, and the step count is the real bound.

Measured over the six acceptance runs: five desks finished in 1.9 to 5.2 minutes, and
the Tesla valve -- the request with the most clauses in it -- was still working at
fourteen. A lap is 30-40 s and a meshing command can be minutes, so at fifteen the
clock was cutting a run whose own step budget was nowhere near spent. What should stop
a run is thirty commands without an answer, not a mesh that takes three minutes to
build."""

STEP_TIMEOUT_S = 240
"""One command. Longer work goes in the background and is polled -- the brief says so."""

MAX_REPLY_TOKENS = 8000
OUTPUT_CHARS = 6000
"""What comes back from one command. Mesh logs are long and repetitive, and the news is
at both ends -- the command that failed and the summary that followed -- so a long
output is cut in the middle rather than truncated."""

TRANSCRIPT_ROWS = 40
"""How far back into the session's transcript to look for the person's own words."""

SAID_LINES = 6
"""How many of them travel with the job. The last few are the ones that are about
this mesh; a whole session's worth would bury the request in an older study's."""

RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}
"""Model-API failures worth one more try: overloaded, rate-limited, gateway. A 400 or
a 401 says the same thing twice."""

RETRY_PAUSE_S = 5.0

KEEP_IMAGES = 2
"""Pictures kept in the thread. Older observations keep their words and lose their
image: a render is 800-1300 tokens and the fifth-oldest one is answering a question
that was settled three steps ago."""

_FENCE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)
_PNG = re.compile(r"[\w./~-]*\.png\b")


@dataclass
class Step:
    """One lap, as the record keeps it."""

    cmd: str
    exit_code: int
    seconds: float
    output: str = ""
    image: str = ""
    """Workspace path of the picture that came back, when one did."""


@dataclass
class MeshResult:
    ok: bool = False
    case_rel: str = ""
    case_dir: str = ""
    summary: str = ""
    """The agent's own closing words -- what it built and what it did not check."""
    check: Check | None = None
    png: bytes | None = None
    steps: list[Step] = field(default_factory=list)
    seconds: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)
    error: str = ""
    remarks: list[str] = field(default_factory=list)
    """What the person said while this ran, in their words. Reported back so the
    calling agent is not the last to hear about a change it did not make."""
    stopped: str = ""
    """Empty when it finished on its own terms; else 'steps', 'time' or 'provider'."""


class Mesher:
    """The mesh desk, wired to a workspace and a model.

    One instance per session; `run` is one geometry. It holds no state between runs
    beyond the provider client, so a second call starts from a clean thread.
    """

    def __init__(self, cfg: Any, backend: Any, store: Any, home: str,
                 on_step: Callable[[Step], None] | None = None,
                 interject: Callable[[], str | None] | None = None):
        self.cfg = cfg
        self.backend = backend
        self.store = store
        """The session's transcript, read for what the person actually said."""
        self.home = (home or backend.workspace_root).rstrip("/")
        self.on_step = on_step
        self.interject = interject
        """Drains anything the person has typed since the last call, or None.

        The same callable the main loop uses between its own tool calls. Held here as
        well because this desk holds the thread for minutes at a time, and a remark
        that waits that long is a remark that arrives after the thing it was about."""
        self.provider = make_provider(cfg)
        self.model = cfg.mesher_model or cfg.model
        self.effort = cfg.mesher_effort or "high"
        self.max_steps = int(cfg.mesher_max_steps or MAX_STEPS)
        self.max_seconds = float(cfg.mesher_max_seconds or MAX_SECONDS)

    # -- the run ---------------------------------------------------------------

    def run(self, request: str, case: str | None = None) -> MeshResult:
        case_rel = _case_name(case)
        case_dir = f"{self.home}/{case_rel}"
        started = time.monotonic()
        self._sent: dict[str, tuple[int, int]] = {}
        result = MeshResult(case_rel=case_rel, case_dir=case_dir)
        try:
            self.backend.exec(f"mkdir -p {shlex.quote(case_dir)}", timeout_s=60)
        except Exception as exc:  # noqa: BLE001 - the workspace answered badly; say so
            result.error = f"could not make the case directory: {exc}"
            result.seconds = time.monotonic() - started
            return result

        system = system_prompt(STEP_TIMEOUT_S)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text",
                                          "text": task_message(request, case_dir, case_rel,
                                                               self._said())}]}
        ]
        last_text = ""
        turns = 0
        """Every lap counts against the budget, not only the ones that ran a command:
        a desk that answers with prose, or says done to a check that refuses it, is
        spending the same minutes and would otherwise loop until the clock."""

        while True:
            if turns >= self.max_steps:
                result.stopped = "steps"
                break
            if time.monotonic() - started >= self.max_seconds:
                result.stopped = "time"
                break
            try:
                turn = self._turn(system, messages)
            except ProviderError as exc:
                result.error = f"the model call failed: {exc}"
                result.stopped = "provider"
                break
            turns += 1
            _add(result.tokens, turn.tokens)
            said = _assistant(turn)
            if said is None:
                # A turn that is all reasoning and no words. Sending it back verbatim
                # is a 400 from the Messages API ("text content blocks must be
                # non-empty") which killed a whole run mid-mesh, so the empty turn is
                # dropped and the desk is asked for the command it did not send.
                _observe(messages, "That message arrived empty. Send one fenced ```bash "
                                   "block with the command you want run.")
                continue
            messages.append(said)
            last_text = turn.text.strip() or last_text

            remark = self._remark(messages, result)

            cmd, complaint = parse_action(turn.text)
            if complaint:
                _observe(messages, complaint)
                continue

            if _is_finish(cmd) and remark:
                # Somebody spoke in the same breath as "done". Their words are the
                # newer instruction, so the run continues rather than closing on a
                # shape that was right one message ago.
                continue

            if _is_finish(cmd):
                result.summary = _summary(turn.text)
                check = verify(self.backend, case_dir, case_rel, request)
                result.check = check
                if check.ok:
                    result.ok = True
                    break
                _observe(messages, check.as_refusal())
                continue

            step = self._exec(cmd, case_dir, messages)
            result.steps.append(step)
            if self.on_step:
                try:
                    self.on_step(step)
                except Exception:  # noqa: BLE001 - a progress line may not end a run
                    pass
            _evict(messages)

        result.seconds = time.monotonic() - started
        if not result.ok and result.check is None:
            # The run ended without saying done -- out of steps, out of time, or the
            # model call failed. What is on disk may still be a finished mesh, and a
            # mesh nobody looked at is exactly the failure this desk exists to end.
            # Measured, not assumed: a T10 run built 91,000 cells, hit a 400 on its
            # next model call, and was reported as "nothing was meshed".
            result.check = verify(self.backend, case_dir, case_rel, request)
            result.ok = result.check.ok
        if not result.summary:
            result.summary = _summary(last_text)
        result.png = self._render_bytes(result)
        return result

    # -- the person ------------------------------------------------------------

    def _said(self) -> list[str]:
        """The last few things the person typed in this session, verbatim.

        Read off the transcript the session is already writing, so nothing new has to
        be plumbed through the calling agent, and what reaches the desk is the person's
        own words rather than a paraphrase of them.
        """
        try:
            rows = self.store.recent_messages(TRANSCRIPT_ROWS)
        except Exception:  # noqa: BLE001 - no transcript is not a reason not to mesh
            return []
        said = [" ".join(str(row.get("content") or "").split())
                for row in rows if row.get("role") == "user"]
        return [line[:600] for line in said if line][-SAID_LINES:]

    def _remark(self, messages: list[dict[str, Any]], result: "MeshResult") -> str:
        """Anything the person has typed since the last step, put into the thread."""
        if self.interject is None:
            return ""
        try:
            text = self.interject()
        except Exception:  # noqa: BLE001 - a remark that cannot be read may not end a run
            return ""
        if not (text or "").strip():
            return ""
        _observe(messages, remark_message(text))
        result.remarks.append(text.strip())
        return text.strip()

    def _turn(self, system: str, messages: list[dict[str, Any]]) -> Any:
        """One model call, retried once when the failure is one that passes.

        A desk five minutes into a mesh is expensive to lose to an overloaded endpoint,
        and the run cannot resume: the next call starts a clean thread. A 400 is not
        retried -- it is a fact about the request, and repeating it costs the same
        error twice."""
        for attempt in (1, 2):
            try:
                return self.provider.stream(
                    model=self.model, system=system, messages=messages, tools=[],
                    effort=self.effort, max_tokens=MAX_REPLY_TOKENS, listener=Listener(),
                )
            except ProviderError as exc:
                if attempt == 2 or exc.status_code not in RETRY_STATUSES:
                    raise
                time.sleep(RETRY_PAUSE_S)
        raise AssertionError("unreachable")

    # -- the machine -----------------------------------------------------------

    def _exec(self, cmd: str, case_dir: str, messages: list[dict[str, Any]]) -> Step:
        t0 = time.monotonic()
        try:
            outcome = self.backend.exec(cmd, cwd=case_dir, timeout_s=STEP_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - the workspace, not the command
            seconds = time.monotonic() - t0
            _observe(messages, f"the command could not be run: {exc}")
            return Step(cmd=cmd, exit_code=-1, seconds=seconds, output=str(exc))
        seconds = time.monotonic() - t0
        output = outcome.output or ""
        if outcome.job_id:
            output = (f"{output}\nThis outran the {STEP_TIMEOUT_S} s window and is now "
                      f"running detached as job {outcome.job_id}. Poll its log with "
                      "`tail`; do not start it again.")
        body = _clip(output, OUTPUT_CHARS)
        head = f"exit {outcome.exit_code} ({seconds:.0f} s)"
        if outcome.truncated and outcome.log_path:
            head += f"; output cut, all of it is at {outcome.log_path}"
        blocks: list[dict[str, Any]] = [{"type": "text", "text": f"{head}\n{body}".rstrip()}]

        path, data = self._picture(cmd, case_dir)
        if data:
            blocks.append(images.attachment(images.downscale(data, "image/png"), "image/png"))
        messages.append({"role": "user", "content": blocks})
        return Step(cmd=cmd, exit_code=outcome.exit_code, seconds=seconds,
                    output=body, image=path)

    def _picture(self, cmd: str, case_dir: str) -> tuple[str, bytes | None]:
        """The PNG this command wrote, if it wrote one it named.

        No protocol and nothing for the model to remember: it renders, and the render
        arrives. Only the last path named is fetched -- a command that draws four
        panels into one file is the normal case, and four separate pictures a step is
        a token bill nobody asked for.

        A file that has not changed since it was last sent is not sent again. A command
        that merely mentions an old render (a `cat` of a script, an `ls`) would
        otherwise put yesterday's picture in front of the desk as if the command had
        just drawn it -- which is the exact mistake this whole segment exists to stop,
        in the one place that is the harness's fault rather than the model's.
        """
        for name in reversed(_PNG.findall(cmd)):
            path = name if name.startswith("/") else f"{case_dir}/{name.lstrip('./')}"
            try:
                info = self.backend.stat(path)
            except Exception:  # noqa: BLE001 - not there is the common case
                continue
            if info.is_dir or not info.size or info.size > images.MAX_ATTACH_BYTES:
                continue
            if self._sent.get(path) == (info.size, info.mtime):
                continue
            try:
                data = self.backend.get_file(path, limit=info.size)
            except Exception:  # noqa: BLE001
                continue
            if len(data) == info.size:
                self._sent[path] = (info.size, info.mtime)
                return path, data
        return "", None

    def _render_bytes(self, result: MeshResult) -> bytes | None:
        """The picture the tool result carries back: the one the check drew."""
        path = (result.check.render_abs or result.check.render) if result.check else ""
        if not path:
            return None
        try:
            info = self.backend.stat(path)
            if info.size and info.size <= images.MAX_ATTACH_BYTES:
                data = self.backend.get_file(path, limit=info.size)
                return data if len(data) == info.size else None
        except Exception:  # noqa: BLE001 - a missing picture is not a failed mesh
            return None
        return None


# -- reading what the model sent ----------------------------------------------


def parse_action(text: str) -> tuple[str, str]:
    """The one command in a message, or what to say back about it.

    Returns `(cmd, "")` or `("", complaint)`. Both halves matter: a model that wrote
    two blocks has to be told which one would have run, and a model that wrote none is
    usually explaining itself at length instead of acting.
    """
    blocks = [b.strip() for b in _FENCE.findall(text or "") if b.strip()]
    if not blocks:
        return "", ("There was no bash block in that message, so nothing ran. Send one "
                    "fenced ```bash block containing the command you want run, and "
                    "nothing else that needs running.")
    if len(blocks) > 1:
        return "", (f"That message had {len(blocks)} bash blocks and none of them ran. "
                    "One block per message: put the whole command -- a heredoc, a "
                    "pipeline, `&&` -- in a single block.")
    return blocks[0], ""


_FINISH = re.compile(rf"^echo\s+[\"']?{MESH_DONE}[\"']?$")


def _is_finish(cmd: str) -> bool:
    """Whether this block *is* the finish command, not whether it mentions it.

    It used to be a search for the word anywhere in the block, and a model that
    writes `# echo MESH_DONE once checkMesh passes` above a perfectly ordinary
    command -- which is a very normal thing to write, since the brief hands it the
    token -- had that command silently never run, and got the finish check's refusal
    as the answer to something it had not asked. The token is a whole command or it
    is a word in a comment.
    """
    return bool(_FINISH.match(cmd.strip()))


def _summary(text: str) -> str:
    """The prose around the block: what the agent says it built."""
    without = _FENCE.sub("", text or "").strip()
    return "\n".join(line for line in without.splitlines() if line.strip()).strip()


# -- the thread ---------------------------------------------------------------


def _assistant(turn: Any) -> dict[str, Any] | None:
    """The turn as a thread entry, with empty blocks left out, or None if nothing is left.

    An empty text block is not a harmless nothing: the Messages API refuses the whole
    request that carries one, so a single empty turn ends the run several steps into a
    mesh that was going fine.
    """
    message = turn.as_message()
    content = [b for b in (message.get("content") or []) if not _is_empty(b)]
    if not content:
        return None
    message["content"] = content
    return message


def _is_empty(block: Any) -> bool:
    kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
    if kind != "text":
        return False
    text = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
    return not (text or "").strip()


def _observe(messages: list[dict[str, Any]], text: str) -> None:
    messages.append({"role": "user", "content": [{"type": "text", "text": text}]})


def _evict(messages: list[dict[str, Any]]) -> None:
    """Keep the last `KEEP_IMAGES` pictures; older ones become a line of text.

    The words stay, because the words are what the later steps reason from -- the
    measurements, the patch table, the checkMesh line. Only the picture goes.
    """
    seen = 0
    for message in reversed(messages):
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            continue
        blocks = message["content"]
        if not any(isinstance(b, dict) and b.get("type") == "image" for b in blocks):
            continue
        seen += 1
        if seen <= KEEP_IMAGES:
            continue
        kept = [b for b in blocks if not (isinstance(b, dict) and b.get("type") == "image")]
        kept.append({"type": "text", "text": "(the picture from this step is no longer shown)"})
        message["content"] = kept


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    cut = len(text) - limit
    return f"{text[:head]}\n... {cut} characters cut ...\n{text[-tail:]}"


def _add(total: dict[str, int], more: dict[str, int]) -> None:
    for key, value in (more or {}).items():
        total[key] = total.get(key, 0) + int(value or 0)


def _case_name(case: str | None) -> str:
    """A directory name under the study, never a path out of it."""
    name = (case or "").strip().strip("/")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return name or "mesh"
