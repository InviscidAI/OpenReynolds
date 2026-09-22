"""An independent look at the finished mesh, by a model that did not build it.

The finish line the desk cannot declare for itself is `check.py`: a bare `checkMesh`
per region, and the advisory probes. All of it is about the mesh as a mesh. None of it
is about the *shape* -- whether what was built is what was asked for -- and the only
one who ever looked at the shape was the desk, in its own thread, at renders it drew
of a geometry it had just reasoned into existence. That is the self-confirmation the
sweeps keep recording: `core-postmerge-20260921` scored 24/26 green and the human vet
passed 21, and the three that got through -- a missing trailing-edge wall (T10), a
named clearance spanned by one cell (T9, T19) -- passed `checkMesh` and every probe.

So this is a second pair of eyes with nothing in front of them but the request and
the pictures. `Reviewer.review` renders the mesh from several views over the backend
(`toolbox/mesh_look.py --views`, staged into the case the way the advisory gate stages
its scripts), opens a **fresh thread** -- no history, no reasoning, none of the desk's
own words except its closing summary -- and asks one question through one tool:
does this pass? The answer arrives as a `verdict` tool call, because a tool schema is
the one structured-output channel every provider here speaks.

Three rules keep it from being the nitpicker a reviewer becomes by default:

* **the decision is the harness's, not the model's.** `Review.blocks_finish` is True
  only for a `fail` with at least one `blocking` problem *and* a confidence the model
  would bet on. A note, an unsure fail, a render that could not be made -- none of
  these stop the desk. What stops it is a confident, named defect in a named view;
* **it fails for what changes the CFD answer**, and the brief says so at length: wrong
  topology, wrong shape class, proportions off by a fifth, the wrong ends for inlet and
  outlet. Mesh density, `checkMesh` numbers and choices the request left open are not
  its business, and it is told to say which view would settle anything it cannot see;
* **it is bounded.** `MAX_ROUNDS` fail verdicts per desk run, one retry on the model
  call, one re-ask when the reply carried no tool call, and every failure of its own
  machinery is `skipped` -- never a pass, never a fail, and never an exception into the
  middle of a declare.

Nothing here opens a mesh or a CAD kernel. The rendering runs where the mesh is and
the pictures come back as bytes, which is the same rule `check.py` lives by.
"""

from __future__ import annotations

import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .. import images
from ..llm import Listener, ProviderError, make_provider
from .check import VIEW_SCRIPTS, _json_in, staged


MAX_ROUNDS = 2
"""How many times one desk run may be failed by the reviewer.

Two, because the loop it closes has to close: the desk fixes what it is told or says
why the reviewer is wrong, and a third declare is accepted with the reviewer's
concerns reported upward rather than argued about forever. Every round is a render,
a model call with three pictures in it, and another turn of the desk."""

REVIEW_TIMEOUT_S = 180
"""Render plus fetch. The render is `mesh_look.py` opening the mesh once and drawing
three composites; on the meshes this desk builds that is seconds, and a minute on a
few million cells. `checkMesh` is skipped -- it has already been judged."""

VIEW_EDGE = 1024
"""Longest edge a view is scaled to before it is attached. `images.ATTACH_MAX_EDGE`'s
argument: a mesh at 1024 px shows everything a shape check is looking for at half the
tokens of the render as drawn."""

MAX_VIEWS = 3
"""Composite PNGs per review, and what `mesh_look.py --views` writes. Bounded so a
review costs three or four thousand image tokens, not thirty."""

MAX_REPLY_TOKENS = 4000
"""A verdict is a tool call with a few sentences in it. Thinking shares this budget
(`cad/agent.MAX_REPLY_TOKENS` says why), and a reviewer that needs more than this to
decide is not sure -- which is a pass with notes, by the rule below."""

RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}
"""Model-API failures worth one more try. The same set as `cad/agent.RETRY_STATUSES`,
written here rather than imported so the desk can import this module lazily without
the two importing each other."""

RETRY_PAUSE_S = 5.0

CONFIDENT = ("sure", "likely")
"""The confidences that let a fail block the finish. `unsure` is a pass with notes:
the brief tells the reviewer that only what it would bet on is a fail, and this is
where that sentence is enforced rather than hoped for."""

VERDICT_NAME = "verdict"

VERDICT_TOOL: dict[str, Any] = {
    "name": VERDICT_NAME,
    "description": (
        "Deliver your review. This is the only way to answer: prose without this call "
        "is not a verdict and you will be asked again. `pass` means the mesh is what "
        "was asked for as far as these views show; `fail` means you see something that "
        "would change the CFD answer and you would bet on it. Every problem names the "
        "view it is seen in and why it matters to the flow; `blocking` is for what "
        "changes the answer, `note` for everything else worth saying. `confidence` is "
        "yours to state honestly -- an `unsure` fail is treated as a pass with notes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail"],
                "description": "pass, or fail for a defect you would bet on.",
            },
            "confidence": {
                "type": "string",
                "enum": ["sure", "likely", "unsure"],
                "description": (
                    "How sure you are of the verdict as a whole. `sure`: the views "
                    "show it plainly. `likely`: you would bet on it. `unsure`: you "
                    "suspect it, but a view you do not have would be needed to settle "
                    "it -- say which one in the problem."),
            },
            "problems": {
                "type": "array",
                "description": (
                    "What is wrong or worth noting, one entry each. Empty on a clean "
                    "pass."),
                "items": {
                    "type": "object",
                    "properties": {
                        "what": {
                            "type": "string",
                            "description": "The defect, in one sentence.",
                        },
                        "seen_in": {
                            "type": "string",
                            "description": (
                                "Which view shows it, by file name, and where in "
                                "that view."),
                        },
                        "why_it_matters": {
                            "type": "string",
                            "description": (
                                "What it does to the CFD answer, in one sentence."),
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["blocking", "note"],
                            "description": (
                                "`blocking` changes the answer; `note` is worth "
                                "saying and does not."),
                        },
                        "fixed": {
                            "type": "string",
                            "enum": ["fixed", "not fixed", ""],
                            "description": (
                                "On a re-review only: whether this problem from the "
                                "last review is now fixed. Leave empty for a new "
                                "problem."),
                        },
                    },
                    "required": ["what", "seen_in", "why_it_matters", "severity"],
                },
            },
            "looked_fine": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "What you checked and found right, briefly -- so the reader knows "
                    "what was looked at, not only what was found."),
            },
        },
        "required": ["verdict", "confidence"],
    },
}
"""The structured-output channel.

A tool rather than a JSON-in-prose convention for the reason `cad/agent.CELL_TOOL` gives:
the API enforces the shape, the turn stops on the call, and a model that wants to
explain itself at length can do so in the text beside it without the explanation being
mistaken for the answer. `checkMesh` is not in here anywhere, on purpose -- it has been
judged, and a reviewer that could relitigate it would.
"""

REVIEW_SYSTEM = """You are an independent reviewer of a finished CFD mesh. You did not \
build it and you have not seen the conversation that did. In front of you are the \
request that was made, the words of the person who made it, what the builder says it \
built, some measured facts about the mesh, and several pictures of the mesh. Your job \
is to compare the pictures to the request and say whether this is the geometry that was \
asked for. Nothing else.

Look at every view before you decide. Name the view each problem is seen in, and where \
in it. What each view shows:

For a 2D (one cell thick) case, three pictures:
- overview.png -- the boundary patches, one colour per patch with a legend, looking \
down z; the front and back `empty` patches are left out because they would cover \
everything.
- cells.png -- every cell, looking down z. This is the shape as the solver has it.
- detail.png -- a 2x2 grid of zoomed quadrants of the cell view. This is where a curve \
that was built as a polygon, a corner that should be a fillet, or a gap one cell wide \
shows itself.

For a 3D case, three pictures:
- iso.png -- a 2x2 grid of four isometric corners, the boundary patches coloured with a \
legend and the enclosure (a flow box round a body) drawn faint so the body shows.
- ortho.png -- a 2x3 grid of the six axis views: +x, -x, +y, -y, +z, -z.
- cuts.png -- a 2x2 grid of cuts through the cells: through the middle in x, y and z, \
and one off-centre.

FAIL only for what would change the CFD answer. That is:
- wrong topology: a feature missing or extra, a channel that does not connect, the \
wrong number of loops, fins, passages, holes or bodies;
- wrong shape class: polygonal or sharp-cornered where the request implies smooth \
curves -- the lobes of a Tesla valve, the nose of an aerofoil, a bend, a fillet, a \
circle built as a hexagon;
- proportions off by more than about 20% of what the request states or clearly implies \
-- a length, a diameter, a spacing, an angle;
- a 2D case that is not one cell thick, or a 3D case that is flat;
- inlet and outlet on visibly the wrong ends, or the wrong faces;
- a named gap or clearance spanned by a single cell -- a passage the request cares \
about that the mesh cannot resolve is a passage that is not there;
- a symmetry the request implies that is visibly broken.

NOT a fail, whatever you think of it:
- mesh density, grading, cell shape or how the mesh looks. checkMesh has already judged \
the mesh as a mesh, and its numbers are not yours to relitigate;
- anything the request left unstated. The builder had to choose, and a choice you \
would have made differently is a note at most;
- rendering artefacts: a legend over a patch, a faint enclosure, a cut plane that \
happens to fall between cells, aliasing;
- anything you cannot see in these views. Say which view would settle it, as a note.

Only fail if you would bet on it. If you suspect something and cannot see it clearly, \
say so as a note with confidence `unsure` -- an unsure fail is treated as a pass with \
notes, so you lose nothing by being honest and the builder loses a round if you are \
not. A `blocking` problem is one that changes the answer; a `note` is anything else \
worth saying. Say what looked fine, briefly, so the reader knows what you checked.

If this is a re-review, you will be told what you flagged last time. For each of those, \
say `fixed` or `not fixed` in its `fixed` field, with the view that shows it. Add a new \
blocking problem only if it would be negligent to miss -- the point of a re-review is \
to close the loop, not to open another.

Answer through the `verdict` tool, and only through it. Prose beside the call is \
welcome; prose instead of it is not an answer."""


# -- what a review is ----------------------------------------------------------


@dataclass
class Problem:
    """One thing the reviewer saw, in the register the desk is handed it in."""

    what: str
    seen_in: str
    why_it_matters: str
    severity: str = "note"
    """`blocking` or `note`. Anything the model said that is not `blocking` is a note:
    the schema offers two words and a third is a note by construction."""
    fixed: str = ""
    """On a re-review: `fixed`, `not fixed`, or empty for a problem raised fresh."""

    @property
    def blocking(self) -> bool:
        return self.severity == "blocking"

    def bullet(self) -> str:
        where = f" (seen in {self.seen_in})" if self.seen_in else ""
        why = f": {self.why_it_matters}" if self.why_it_matters else ""
        state = f" [{self.fixed}]" if self.fixed else ""
        return f"- {self.what}{where}{why}{state}"

    @classmethod
    def from_dict(cls, data: Any) -> "Problem":
        if not isinstance(data, dict):
            return cls(what=str(data), seen_in="", why_it_matters="")
        severity = str(data.get("severity") or "").strip().lower()
        fixed = str(data.get("fixed") or "").strip().lower()
        return cls(
            what=str(data.get("what") or "").strip(),
            seen_in=str(data.get("seen_in") or "").strip(),
            why_it_matters=str(data.get("why_it_matters") or "").strip(),
            severity="blocking" if severity == "blocking" else "note",
            fixed=fixed if fixed in ("fixed", "not fixed") else "",
        )


@dataclass
class Review:
    """What the reviewer said, and what the harness makes of it.

    `verdict` is the model's word; `blocks_finish` is the harness's decision, and the
    two are deliberately not the same thing.
    """

    verdict: str = "skipped"
    """`pass`, `fail`, or `skipped` -- the last for every failure of the review's own
    machinery: no render, no model, no tool call. Never a verdict about the mesh."""
    confidence: str = ""
    """`sure`, `likely` or `unsure`, as the model said it."""
    problems: list[Problem] = field(default_factory=list)
    looked_fine: list[str] = field(default_factory=list)
    views: list[str] = field(default_factory=list)
    """The pictures it was shown, as paths relative to the case."""
    png: bytes | None = None
    """The first view, downscaled -- what the tool result carries back to the main
    agent when the desk itself drew nothing."""
    seconds: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)
    round: int = 1
    """Which review of this desk run this is; `as_work` says it to the desk."""
    error: str = ""
    """Why it was skipped, when it was."""
    unresolved: bool = False
    """The rounds ran out on a fail: the declare was accepted and this review is the
    concern the main agent and the person are handed."""
    summary: str = ""
    """The model's prose beside the tool call, if any."""

    @property
    def blocking(self) -> list[Problem]:
        return [p for p in self.problems if p.blocking]

    @property
    def notes(self) -> list[Problem]:
        return [p for p in self.problems if not p.blocking]

    def blocks_finish(self) -> bool:
        """The harness-side decision rule, and the whole of it.

        A fail with a blocking problem the reviewer would bet on. Notes do not block, an
        unsure fail does not block, a skipped review does not block. This is what keeps
        a second opinion from becoming a veto over taste."""
        return (self.verdict == "fail" and bool(self.blocking)
                and self.confidence in CONFIDENT)

    def as_work(self, round: int, max_rounds: int) -> str:
        """What the desk is told when the review did not pass: the problems as work."""
        head = (f"An independent reviewer looked at the mesh in {len(self.views)} "
                f"views and did not pass it (review {round} of {max_rounds}):")
        parts = [head]
        parts.extend(p.bullet() for p in self.blocking)
        if self.notes:
            parts.append("Also noted (not blocking):")
            parts.extend(p.bullet() for p in self.notes)
        parts.append("Fix what is wrong, or if the reviewer is mistaken say why in your "
                     "closing summary, then declare again.")
        return "\n".join(parts)

    def lines(self) -> list[str]:
        """The report the main agent and the person read, after the mesh's own."""
        if self.verdict == "skipped":
            return [f"not reviewed: {self.error or 'no reason recorded'}"]
        if self.unresolved:
            out = [f"independent review: did not pass after {self.round} rounds -- "
                   "treat this mesh as suspect:"]
            out.extend(f"  {p.bullet()}" for p in self.problems)
            return out
        if self.blocks_finish():
            out = [f"independent review: FAIL -- looked at {len(self.views)} views:"]
            out.extend(f"  {p.bullet()}" for p in self.problems)
            return out
        out = [f"independent review: PASS -- looked at {len(self.views)} views"]
        if self.verdict == "fail":
            # A fail the harness did not act on: unsure, or notes only. The concern is
            # still reported, because a low-confidence fail is information, and the
            # word beside it says why it did not bind.
            out.append(f"  (the reviewer said fail with confidence "
                       f"{self.confidence or 'unstated'} and no blocking problem it "
                       "would bet on, which does not bind)")
        for problem in self.problems:
            out.append(f"  notes: {problem.bullet()[2:]}")
        return out


# -- the facts beside the pictures ------------------------------------------------

FACT_KEYS = ("cells", "faces", "points", "bounds", "two_d", "patches", "checkmesh",
             "regions")


def facts_from_check(check: Any) -> dict[str, Any]:
    """The measured facts off a `cad.check.Check`, as a plain dict.

    Duck-typed with defaults, because the reviewer is also run standalone by the
    `mesh_review` tool against a case nobody has checked, and a payload straight out of
    `mesh_look.py` is the other thing that fills this shape."""
    facts: dict[str, Any] = {}
    for key in FACT_KEYS:
        value = getattr(check, key, None)
        if value is None and isinstance(check, dict):
            value = check.get(key)
        facts[key] = value
    facts["cells"] = int(facts.get("cells") or 0)
    facts["faces"] = int(facts.get("faces") or 0)
    facts["points"] = int(facts.get("points") or 0)
    facts["bounds"] = [float(v) for v in (facts.get("bounds") or [])]
    facts["two_d"] = bool(facts.get("two_d"))
    facts["patches"] = [dict(p) for p in (facts.get("patches") or []) if isinstance(p, dict)]
    facts["checkmesh"] = str(facts.get("checkmesh") or "")
    facts["regions"] = [str(r) for r in (facts.get("regions") or []) if r]
    return facts


def facts_text(facts: dict[str, Any] | None) -> str:
    """The facts as the reviewer reads them: numbers beside the pictures.

    Bounds in metres and, when the span is under 10 m, in millimetres too -- the
    request was probably written in them, and a mesh a thousand times too big is the
    one defect no picture shows."""
    facts = facts_from_check(facts or {})
    out: list[str] = []
    if facts["cells"]:
        shape = "2D, one cell thick" if facts["two_d"] else "3D"
        counts = f"{facts['cells']:,} cells"
        if facts["faces"]:
            counts += f", {facts['faces']:,} faces"
        if facts["points"]:
            counts += f", {facts['points']:,} points"
        out.append(f"{counts} ({shape})")
    else:
        out.append("2D, one cell thick" if facts["two_d"] else "3D")
    if len(facts["regions"]) > 1:
        out.append("regions: " + ", ".join(facts["regions"]))
    bounds = facts["bounds"]
    if len(bounds) == 6:
        x0, y0, z0, x1, y1, z1 = bounds
        span = (x1 - x0, y1 - y0, z1 - z0)
        line = (f"bounds {span[0]:.4g} x {span[1]:.4g} x {span[2]:.4g} m "
                f"(x {x0:.4g}..{x1:.4g}, y {y0:.4g}..{y1:.4g}, z {z0:.4g}..{z1:.4g})")
        if 0 < max(span) < 10:
            line += (f" = {span[0] * 1000:.4g} x {span[1] * 1000:.4g} x "
                     f"{span[2] * 1000:.4g} mm")
        out.append(line)
    if facts["checkmesh"]:
        out.append(f"checkMesh: {facts['checkmesh']} (already judged; not yours to re-decide)")
    if facts["patches"]:
        out.append("patches:")
        for patch in facts["patches"]:
            bits = [f"  {patch.get('name', '?')}", str(patch.get("type", "?")),
                    f"{int(patch.get('nFaces') or 0):,} faces"]
            if patch.get("area"):
                bits.append(f"area {float(patch['area']):.4g} m2")
            normal = patch.get("normal")
            if normal and len(normal) == 3:
                bits.append(f"normal ({normal[0]:+.2f} {normal[1]:+.2f} {normal[2]:+.2f})")
            if patch.get("region"):
                bits.append(f"in {patch['region']}")
            out.append(" ".join(bits))
    return "\n".join(out)


# -- the reviewer ----------------------------------------------------------------


class _CannotRender(Exception):
    """The pictures could not be made or fetched. Never a verdict about the mesh."""


class Reviewer:
    """Renders a case, shows it to a fresh model with the request, reads the verdict.

    One instance per session, like the desk; `review` is one look. It holds nothing
    between looks but the provider client, so every review is a clean thread.
    """

    def __init__(self, cfg: Any, backend: Any,
                 on_step: Callable[[Any], None] | None = None):
        self.cfg = cfg
        self.backend = backend
        self.on_step = on_step
        """Told one synthetic `Step` per review, so the progress line that renders the
        desk's cells shows the review too. Reports; never consulted."""
        self.model = cfg.review_model or cfg.mesher_model or cfg.model
        """`OPENREYNOLDS_REVIEW_MODEL`, else the desk's model. A different model is the
        interesting arm of the experiment; the same model in a fresh thread is the
        cheaper one, and still not the same eyes that built it."""
        self.effort = cfg.mesher_effort or "high"
        self.provider = make_provider(cfg)
        self.enabled = True

    # -- the look ----------------------------------------------------------------

    def review(self, case_dir: str, case_rel: str, request: str,
               said: list[str] | None, summary: str, facts: dict[str, Any] | None,
               prior: Review | None = None) -> Review:
        """Render, ask, read. Never raises: every failure of its own is `skipped`."""
        started = time.monotonic()
        review = Review(round=(prior.round + 1) if prior else 1)
        try:
            views, pngs, payload = self._render(case_dir)
        except Exception as exc:  # noqa: BLE001 - not a verdict about the mesh
            review.error = f"could not render the mesh for review: {exc}"
            review.seconds = time.monotonic() - started
            self._report(review)
            return review
        review.views = views
        review.png = pngs[0] if pngs else None
        if facts is None:
            facts = facts_from_check(payload)

        messages = [{"role": "user", "content": self._content(
            case_rel, request, said, summary, facts, prior, views, pngs)}]
        try:
            turn, call = self._ask(messages, review.tokens)
        except ProviderError as exc:
            review.error = f"the reviewer's model call failed: {exc}"
            review.seconds = time.monotonic() - started
            self._report(review)
            return review
        except Exception as exc:  # noqa: BLE001 - the reviewer, not the mesh
            review.error = f"the review could not be completed: {type(exc).__name__}: {exc}"
            review.seconds = time.monotonic() - started
            self._report(review)
            return review

        if call is None:
            review.error = "the reviewer gave no verdict"
            review.summary = (turn.text or "").strip() if turn is not None else ""
            review.seconds = time.monotonic() - started
            self._report(review)
            return review

        _fill(review, call.input or {})
        review.summary = (turn.text or "").strip()
        review.seconds = time.monotonic() - started
        self._report(review)
        return review

    # -- the pictures --------------------------------------------------------------

    def _render(self, case_dir: str) -> tuple[list[str], list[bytes], dict[str, Any]]:
        """Draw the views on the workspace and bring them back as bytes.

        `mesh_look.py` is staged into the case for the length of the command and
        removed, on the same argument as the advisory gate's scripts: this desk is
        measured on having no toolbox, so nothing of ours may be left where it works.
        The JSON is read in the same command, because a workspace round trip is the
        expensive part and a render that answers half-way is worse than one that does
        not.
        """
        case_dir = case_dir.rstrip("/")
        with staged(self.backend, case_dir, VIEW_SCRIPTS) as where:
            outcome = self.backend.exec(_render_command(str(where)), cwd=case_dir,
                                        timeout_s=REVIEW_TIMEOUT_S)
        output = getattr(outcome, "output", "") or ""
        head, marker, tail = output.rpartition(_JSON_MARK)
        payload = _json_in(tail if marker else output)
        if payload is None:
            said = (head if marker else output).strip()[-400:]
            raise _CannotRender(
                f"mesh_look.py wrote no readable answer (exit "
                f"{getattr(outcome, 'exit_code', '?')})" + (f": {said}" if said else ""))
        if not payload.get("polymesh", True):
            raise _CannotRender("there is no constant/polyMesh to look at")
        named = [str(p) for p in (payload.get("views") or []) if p][:MAX_VIEWS]
        if not named:
            why = payload.get("error") or "the payload lists no views"
            raise _CannotRender(f"no views were drawn -- {why}")
        views: list[str] = []
        pngs: list[bytes] = []
        for rel in named:
            path = rel if rel.startswith("/") else f"{case_dir}/{rel}"
            try:
                info = self.backend.stat(path)
                size = int(getattr(info, "size", 0) or 0)
                if not size or size > images.MAX_ATTACH_BYTES:
                    continue
                data = self.backend.get_file(path, limit=size)
            except Exception:  # noqa: BLE001 - one missing view is not a failed review
                continue
            if len(data) != size or images.incomplete(data, "image/png"):
                continue
            views.append(rel)
            pngs.append(images.downscale(data, "image/png", VIEW_EDGE))
        if not pngs:
            raise _CannotRender(f"none of the {len(named)} views could be fetched")
        return views, pngs, payload

    # -- the question -------------------------------------------------------------

    def _content(self, case_rel: str, request: str, said: list[str] | None,
                 summary: str, facts: dict[str, Any], prior: Review | None,
                 views: list[str], pngs: list[bytes]) -> list[dict[str, Any]]:
        """The one user message: the words first, then each picture named and shown."""
        parts = [f"Case: {case_rel or 'the case'}.",
                 f"The request, as the calling agent put it:\n{request.strip() or '(none)'}"]
        words = [line.strip() for line in (said or []) if (line or "").strip()]
        if words:
            parts.append("What the person asked for, in their own words (the most "
                         "recent last):\n" + "\n".join(f"> {w}" for w in words))
        parts.append("What the builder says it built (its claims, not measurements "
                     "of yours):\n" + (summary.strip() or "(it said nothing)"))
        parts.append("Measured facts about the mesh:\n" + facts_text(facts))
        if prior is not None:
            flagged = "\n".join(p.bullet() for p in prior.problems) or "- (nothing)"
            parts.append(
                f"This is a re-review (round {prior.round + 1}). Last time you flagged:\n"
                f"{flagged}\n"
                "For each of these say `fixed` or `not fixed` in its `fixed` field, "
                "naming the view that shows it. Add a new blocking problem only if it "
                "would be negligent to miss.")
        parts.append(f"{len(pngs)} pictures follow. Look at every one before answering, "
                     "then answer through the verdict tool.")
        content: list[dict[str, Any]] = [{"type": "text", "text": "\n\n".join(parts)}]
        for index, (rel, png) in enumerate(zip(views, pngs), start=1):
            name = rel.rsplit("/", 1)[-1]
            content.append({"type": "text", "text": f"view {index} of {len(pngs)}: {name}"})
            content.append(images.attachment(png, "image/png"))
        return content

    def _ask(self, messages: list[dict[str, Any]],
             tokens: dict[str, int]) -> tuple[Any, Any]:
        """One question, answered through the tool -- re-asked once if it was not.

        Returns the last turn and the `verdict` call, or `(turn, None)` when two
        replies in a row carried no such call. A tool call by another name is answered
        as an error result, because the API wants every call answered before the
        thread moves on.
        """
        turn = None
        for attempt in (1, 2):
            turn = self._turn(messages)
            _add(tokens, getattr(turn, "tokens", None) or {})
            calls = list(turn.tool_calls or [])
            for call in calls:
                if call.name == VERDICT_NAME:
                    return turn, call
            if attempt == 2:
                break
            assistant = _assistant(turn)
            if assistant is not None:
                messages.append(assistant)
            if calls:
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": c.id, "is_error": True,
                     "content": f"There is no tool called {c.name!r}. Answer through "
                                f"the {VERDICT_NAME} tool."} for c in calls]})
            else:
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": f"Answer through the {VERDICT_NAME} tool."}]})
        return turn, None

    def _turn(self, messages: list[dict[str, Any]]) -> Any:
        """One model call, retried once when the failure is one that passes."""
        for attempt in (1, 2):
            try:
                return self.provider.stream(
                    model=self.model, system=REVIEW_SYSTEM, messages=messages,
                    tools=[VERDICT_TOOL], effort=self.effort,
                    max_tokens=MAX_REPLY_TOKENS, listener=Listener(),
                )
            except ProviderError as exc:
                if attempt == 2 or exc.status_code not in RETRY_STATUSES:
                    raise
                time.sleep(RETRY_PAUSE_S)
        raise AssertionError("unreachable")

    # -- telling whoever is watching -----------------------------------------------

    def _report(self, review: Review) -> None:
        """One synthetic step to the progress line, and nothing if it will not take it."""
        if not self.on_step:
            return
        try:
            from .agent import Step

            first = review.problems[0].what if review.problems else "no problems"
            if review.verdict == "skipped":
                first = review.error or "skipped"
            count = len(review.views)
            self.on_step(Step(
                cmd=f"[review] {review.verdict}: {first}",
                exit_code=1 if review.blocks_finish() else 0,
                seconds=review.seconds,
                output="\n".join(review.lines()),
                image=("1 picture" if count == 1 else f"{count} pictures") if count else "",
                kind="review",
                number=review.round,
                reasoning=review.summary,
            ))
        except Exception:  # noqa: BLE001 - a progress line may not end a review
            pass


# -- helpers ------------------------------------------------------------------------

_JSON_MARK = "@@REVIEW_JSON@@"

VIEWS_REL = "renders/review"
"""Where the views are written in the case. Inside `renders/`, beside the check's own
picture, so the mirror carries them home with everything else."""

LOOK_REL = f"{VIEWS_REL}/look.json"


def _render_command(where: str) -> str:
    """Draw the views, then print the payload after a marker the reader splits on.

    The script's own report goes to the same stream first, so a render that fails
    leaves its complaint where `_render` can quote it."""
    return (f"python3 {shlex.quote(where)}/mesh_look.py . --no-check "
            f"--views {VIEWS_REL} --json {LOOK_REL} 2>&1 | tail -n 30\n"
            f"echo '{_JSON_MARK}'\n"
            f"cat {LOOK_REL} 2>/dev/null\n")


def _fill(review: Review, payload: dict[str, Any]) -> None:
    """The tool call's input into the review, with every word normalised."""
    verdict = str(payload.get("verdict") or "").strip().lower()
    if verdict not in ("pass", "fail"):
        review.verdict = "skipped"
        review.error = f"the reviewer answered with verdict {verdict!r}, which is neither pass nor fail"
        return
    review.verdict = verdict
    review.confidence = str(payload.get("confidence") or "").strip().lower()
    review.problems = [Problem.from_dict(p) for p in (payload.get("problems") or [])
                       if p]
    review.problems = [p for p in review.problems if p.what]
    review.looked_fine = [str(x).strip() for x in (payload.get("looked_fine") or [])
                         if str(x).strip()]


def _assistant(turn: Any) -> dict[str, Any] | None:
    """The turn as a thread entry with empty text blocks left out, or None.

    The same rule as `cad/agent._assistant`, for the same reason: the Messages API
    refuses a request carrying an empty text block."""
    message = turn.as_message()
    content = []
    for block in message.get("content") or []:
        kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
        if kind == "text" and not (text or "").strip():
            continue
        content.append(block)
    if not content:
        return None
    message["content"] = content
    return message


def _add(total: dict[str, int], more: dict[str, int]) -> None:
    for key, value in (more or {}).items():
        total[key] = total.get(key, 0) + int(value or 0)
