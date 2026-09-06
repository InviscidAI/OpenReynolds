"""The geometry desk: a second, specialist loop that authors a shape, draws it, measures
it and commits it -- run where the model is, and handed back to the main agent as one
tool result: the case on the workspace, the picture, the report.

Why a separate loop. The same model one-shots a Tesla-valve mesh in a chat app and, in
this harness, took 66 turns for the wrong shape: every step of authoring a geometry was
a remote round trip (write a script, run it, read the error), and there was no way to
see the shape before meshing. Measured on the fix (`qa-runs/RESULTS-mesh2d.md`): with
`mesh2d.py` in the toolbox the main agent still spends four preview laps of its own
turns getting a shape right, because a spec written first shot is valid five times in
six and correct one time in six -- the model does not reason about where an arc ends or
whether neighbours overlap until it sees the picture and the numbers. Those laps are
the work; this loop does them locally (gmsh and the preview in this process, ~1-2 s a
lap, no sandbox in the path), so from the main agent's point of view a shape is one call.

Why this does not break the free-will contract: as with the front desk (`desk.py`),
the contract governs what may influence the *main model's* decisions. This loop is a
tool with a narrow job; its brief is explicit about its steps, lives here beside it,
and is pinned by `tests/test_geometry_agent.py` rather than by the prompt tests. The
main prompt gains one descriptive sentence naming the tool and nothing that tells the
main model when to use it.

What it never does: run a solver, run Allmesh, touch the main agent's thread or files.
It writes a case directory and returns.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from . import images, trace
from .llm import ProviderError, make_provider
from .llm.base import Listener

TOOLBOX = Path(__file__).resolve().parent / "toolbox"

MAX_LAPS = 6
"""Laps of spec -> picture -> report -> revise before the last good spec is committed."""

MAX_SECONDS = 90.0
"""Wall-clock budget for the laps; the commit itself is not counted."""

MAX_REPLY_TOKENS = 8_000
COMMIT = "COMMIT"

GEOMETRY_SYSTEM = """\
You author a geometry for a CFD case from a request in words, using a spec grammar (given \
in the first message) that a tool turns into a picture and a measured report. You work in \
laps.

Lap 1: reply with ONLY a JSON spec in the grammar -- no prose, no code fences. Lengths in \
metres; a top-level "scale" (for example 0.001) says the numbers are in that unit instead.

Then, each lap, you are shown the picture the tool drew and its report: extent, area, \
islands (enclosed inner loops), each patch's edge count and length, and any warning. \
Compare them with the request. Count the features: islands are enclosed bypasses or \
holes; a passage needs one flat inlet edge and one flat outlet edge; a body in a box has \
inlet, outlet, farfield and body. Check the extent against the sizes asked for. Look, in \
the picture and the report, for open loops, overlaps between neighbours, small pockets, \
notches, slivers, and edges the report warns about.

If anything disagrees, reply with a revised spec (the whole spec, JSON only). If the \
picture and the numbers are the shape that was asked for, reply with the single word \
COMMIT. At most 6 laps. If you cannot make them agree within that, reply COMMIT followed \
by one line starting "disagrees:" saying what still differs, so it is recorded.

Never run a solver, never mesh, never reply with anything but a spec or COMMIT. \
Measure, compare, then commit."""


def grammar(mode: str) -> str:
    """The spec grammar as the tool's own docstring states it."""
    script = TOOLBOX / ("mesh2d.py" if mode == "2d" else "cad_gen.py")
    parts = script.read_text(encoding="utf-8").split('"""', 2)
    return parts[1].strip() if len(parts) >= 2 else ""


def unavailable() -> str | None:
    """Why the desk cannot run in this process, or None when it can."""
    missing = [name for name in ("gmsh", "matplotlib") if importlib.util.find_spec(name) is None]
    if missing:
        return (f"{' and '.join(missing)} not importable in this process (`pip install "
                "openreynolds[geometry]`); on the instance, `mesh2d.py --spec` builds the "
                "same case from a spec written by hand")
    return None


def extract_spec(text: str) -> tuple[Any, bool]:
    """(the JSON spec if the reply carries one, whether the reply is a COMMIT)."""
    stripped = text.strip()
    commit = stripped.upper().startswith(COMMIT)
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.S)
    body = fence.group(1).strip() if fence else stripped
    starts = [i for i in (body.find("{"), body.find("[")) if i >= 0]
    if not starts:
        return None, commit
    try:
        value, _ = json.JSONDecoder().raw_decode(body[min(starts):])
    except json.JSONDecodeError:
        return None, commit
    return value, commit


def build_args(mode: str, spec_path: Path, out_dir: Path, scale: float) -> list[str]:
    """The dry-run + preview call, in this interpreter."""
    script = TOOLBOX / ("mesh2d.py" if mode == "2d" else "cad_gen.py")
    args = [sys.executable, str(script), "--spec", str(spec_path), "--dry-run",
            "--preview", str(out_dir / "preview.png")]
    if scale and scale != 1.0:
        args += ["--scale", str(scale)]
    return args


def case_args(mode: str, spec_path: Path, target: Path, study: str, scale: float) -> list[str]:
    """The commit: the case files, the STEP, the msh (cad_gen) -- no Allmesh, no solver."""
    script = TOOLBOX / ("mesh2d.py" if mode == "2d" else "cad_gen.py")
    args = [sys.executable, str(script), str(target), "--spec", str(spec_path), "--study", study,
            "--preview", str(target / "outline.png"), "--force"]
    if scale and scale != 1.0:
        args += ["--scale", str(scale)]
    return args


class GeometryResult:
    def __init__(self, report: str = "", png: bytes | None = None, case_rel: str = "",
                 laps: int = 0, seconds: float = 0.0, tokens: dict | None = None,
                 agreed: bool = False, disagrees: str = "", error: str = ""):
        self.report = report
        self.png = png
        self.case_rel = case_rel
        self.laps = laps
        self.seconds = seconds
        self.tokens = dict(tokens or {})
        self.agreed = agreed
        self.disagrees = disagrees
        self.error = error


class GeometryAgent:
    """The loop. One provider of its own, the same key and (by default) the same
    model as the main agent; `_client` is the seam a test replaces, as the desk's."""

    def __init__(self, cfg: Any, backend: Any, store: Any, home: str):
        self.backend = backend
        self.store = store
        self.home = str(home or "").rstrip("/")
        self.model = cfg.geometry_model or cfg.model
        self.effort = getattr(cfg, "effort", "medium")
        self.preferences = getattr(cfg, "preferences", "") or ""
        self._provider = make_provider(cfg, timeout=min(120.0, cfg.llm_timeout_s or 120.0))

    @property
    def _client(self):
        return self._provider.client

    @_client.setter
    def _client(self, value) -> None:
        self._provider.client = value

    # -- the parts a test replaces -------------------------------------------

    def _system(self) -> str:
        if not self.preferences.strip():
            return GEOMETRY_SYSTEM
        return (GEOMETRY_SYSTEM + "\n\nThe person keeps a standing note, in their own words:\n"
                + self.preferences.strip())

    def _ask(self, messages: list[dict]) -> Any:
        return self._provider.stream(model=self.model, system=self._system(), messages=messages,
                                     tools=[], effort=self.effort, max_tokens=MAX_REPLY_TOKENS,
                                     listener=Listener())

    def _build(self, mode: str, spec: Any, work: Path, scale: float) -> tuple[int, str, bytes | None]:
        """Dry-run + preview in a child interpreter: a gmsh crash cannot take the runner
        down, and the tool's own report text is the feedback."""
        work.mkdir(parents=True, exist_ok=True)
        spec_path = work / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        try:
            proc = subprocess.run(build_args(mode, spec_path, work, scale), capture_output=True,
                                  text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return 1, "the build ran past 120 s and was stopped", None
        report = (proc.stdout + proc.stderr).strip()
        png = work / "preview.png"
        return proc.returncode, report, (png.read_bytes() if png.exists() else None)

    def _write_case(self, mode: str, spec: Any, study: str, local: Path, scale: float) -> None:
        local.mkdir(parents=True, exist_ok=True)
        spec_path = local / "geometry.json"
        spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        proc = subprocess.run(case_args(mode, spec_path, local, study, scale), capture_output=True,
                              text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError((proc.stdout + proc.stderr).strip()[-1500:] or f"exit {proc.returncode}")

    # -- the loop --------------------------------------------------------------

    def run(self, request: str, mode: str = "2d", study: str = "mesh",
            case: str | None = None) -> GeometryResult:
        reason = unavailable()
        if reason:
            return GeometryResult(error=reason)
        mode = "3d" if str(mode).lower() == "3d" else "2d"
        study = study if study in ("mesh", "steady", "transient") else "mesh"
        case = (case or "geometry").strip("/ ") or "geometry"
        work = Path(tempfile.mkdtemp(prefix="geometry-"))
        messages: list[dict] = [{"role": "user", "content": [{"type": "text", "text": (
            f"The grammar:\n\n{grammar(mode)}\n\nThe request:\n{request.strip()}\n\n"
            "Reply with the JSON spec only.")}]}]
        tokens: dict[str, int] = {}
        laps = 0
        started = time.monotonic()
        last_spec: Any = None
        last_scale = 1.0
        last_report = ""
        last_png: bytes | None = None
        agreed = False
        disagrees = ""

        while laps < MAX_LAPS and time.monotonic() - started < MAX_SECONDS:
            laps += 1
            try:
                turn = self._ask(messages)
            except ProviderError as exc:
                return GeometryResult(error=f"the model call failed: {exc}", laps=laps,
                                      seconds=time.monotonic() - started, tokens=tokens)
            for name, count in (getattr(turn, "tokens", None) or {}).items():
                tokens[name] = tokens.get(name, 0) + int(count)
            text = turn.text
            messages.append(turn.as_message())
            spec, commit = extract_spec(text)
            if commit and last_spec is not None:
                lower = text.lower()
                if "disagrees:" in lower:
                    disagrees = text[lower.index("disagrees:") + len("disagrees:"):].strip().splitlines()[0]
                agreed = not disagrees
                break
            if spec is None:
                messages.append({"role": "user", "content": [{"type": "text", "text": (
                    "That was not a spec. Reply with ONLY the JSON spec in the grammar, or the "
                    "word COMMIT once a built spec is the shape that was asked for.")}]})
                continue
            scale = float(spec.get("scale", 1.0)) if isinstance(spec, dict) else 1.0
            rc, report, png = self._build(mode, spec, work / f"lap{laps}", scale)
            trace.event("geometry_lap", lap=laps, rc=rc, seconds=round(time.monotonic() - started, 1))
            if rc != 0:
                messages.append({"role": "user", "content": [{"type": "text", "text": (
                    f"The tool refused it:\n{report[-2000:]}\n\nReply with a corrected spec.")}]})
                continue
            last_spec, last_scale, last_report, last_png = spec, scale, report, png
            content: list[dict] = []
            if png:
                content.append(images.attachment(images.downscale(png, "image/png"), "image/png"))
            content.append({"type": "text", "text": (
                "The tool drew it (the picture above) and measured:\n" + report[-3000:] +
                "\n\nCompare with the request. Reply with a revised spec, or COMMIT if the "
                "picture and the numbers are the shape that was asked for.")})
            messages.append({"role": "user", "content": content})

        seconds = time.monotonic() - started
        if last_spec is None:
            return GeometryResult(error="no spec built within the laps allowed", laps=laps,
                                  seconds=seconds, tokens=tokens)
        local = Path(self.store.fetch_dir()) / case
        try:
            self._write_case(mode, last_spec, study, local, last_scale)
        except Exception as exc:  # noqa: BLE001 - the report says what happened
            return GeometryResult(report=last_report, png=last_png, laps=laps, seconds=seconds,
                                  tokens=tokens, error=f"the case did not write: {exc}")
        remote = f"{self.home}/{case}" if self.home else case
        self.backend.put_tree(local, remote)
        return GeometryResult(report=last_report, png=last_png, case_rel=remote, laps=laps,
                              seconds=seconds, tokens=tokens, agreed=agreed, disagrees=disagrees)
