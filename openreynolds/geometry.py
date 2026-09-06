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


def _mesh_digest():
    """The toolbox is a directory of scripts, not a package: load the digest by path."""
    spec = importlib.util.spec_from_file_location("toolbox_mesh_digest", TOOLBOX / "mesh_digest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module
TOOLBOX_DEST = "/work/.toolbox"
"""Where the toolbox is refreshed to on the instance (cli.py), for the finish step."""

FINISH_TIMEOUT_S = 240
"""gmshToFoam, checkMesh and one mesh render on a one-cell 2D case take seconds; a 3D
case with a million tets takes about a minute. Past this the finish reports it did not
complete and the main agent has Allmesh to run itself."""

MAX_LAPS = 8
"""Laps of spec -> picture -> report -> revise before the last good spec is committed."""

MAX_SECONDS = 600.0
"""Wall-clock budget for the laps; the commit and the finish are not counted.

Measured on the serpentine run (qa-runs/RESULTS-mesh2d.md): a lap is 30-40 s of model
time at medium effort and 1-2 s of build, and a 90 s budget cut one call at three laps
with every edge still classified `inlet`. At high effort, which is what the desk now
reasons at, the first re-measured valve (2026-09-07, study 20260907-003655-6aab) ran
into a 240 s budget at 246 s: the cap, not agreement, ended it. Ten minutes is the
ceiling the revamp plan sets; the laps end on agreement well before it when the
checks pass, and the main agent sees "still running" meanwhile."""

MAX_REPLY_TOKENS = 16_000
"""The premise gate saw a 2k reply spent entirely on thinking, with no spec in it."""

COMMIT = "COMMIT"

GEOMETRY_SYSTEM = """\
You author a geometry for a CFD case from a request in words, using a spec grammar (given \
in the first message) that a tool turns into a picture and a measured report. You work in \
laps.

Before the first spec, write the request's checkable claims to yourself: every length, \
angle, radius, count and direction it states, and which way the flow goes. The report is \
judged against those, not against the picture looking right.

Lap 1: reply with ONLY a JSON spec in the grammar -- no prose, no code fences. Lengths in \
metres; a top-level "scale" (for example 0.001) says the numbers are in that unit instead. \
Work out where each feature ends before placing the next: a repeated feature's pitch is \
its footprint plus a gap, never less.

Then, each lap, you are shown the picture the tool drew and its report: the measurements \
(extent, area, islands, each patch's edge count and length), a leg table for every \
channel (each leg's start, end and absolute heading, each arc's radius and sweep, where a \
leg lands), and the checks. Compare the leg table with the claims: the angle a branch \
leaves at, the radius of a loop, whether a return leg heads against the flow (a heading \
with a component opposite the passage's axis) are all there as numbers. Count the \
features: islands are enclosed bypasses or holes; a passage has exactly one inlet edge and \
one outlet edge; a body in a box has inlet, outlet, farfield and body. Check the extent \
against the sizes asked for.

A line marked "!! ERROR" means the spec was refused (copies of a repeat that overlap or \
touch, a leg that lands off the body, other than one inlet or outlet): fix that first. A \
line marked "!!" is a warning with coordinates -- a short edge, a channel end read as a \
wall -- and the red crosses on the picture are where they are; fix it or say why it stays.

If anything disagrees with the claims, reply with a revised spec (the whole spec, JSON \
only). If the leg table, the counts and the picture all match the claims and no check \
fails, reply with the single word COMMIT. At most 8 laps. If you cannot make them agree \
within that, reply COMMIT followed by one line starting "disagrees:" saying what still \
differs, so it is recorded.

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
                 agreed: bool = False, disagrees: str = "", error: str = "",
                 script: str = "mesh2d.py", scale: float = 1.0, capped: str = ""):
        self.report = report
        self.png = png
        self.case_rel = case_rel
        self.laps = laps
        self.seconds = seconds
        self.tokens = dict(tokens or {})
        self.agreed = agreed
        self.disagrees = disagrees
        self.error = error
        self.script = script
        """The toolbox script that rebuilds the committed spec on the instance."""
        self.scale = scale
        """The `--scale` that spec was built with (its own top-level "scale", if any)."""
        self.capped = capped
        """"laps" or "time" when the loop ended on a cap rather than on the model's
        COMMIT, so the words can say the last picture was not agreed to."""
        self.meshed = False
        """Whether Allmesh ran to a checkMesh log on the instance (the finish step)."""
        self.mesh_report = ""
        """checkMesh's digest when meshed; else what stopped the finish, in words."""
        self.mesh_png: bytes | None = None
        """The mesh render from the instance, when the finish produced one."""


class GeometryAgent:
    """The loop. One provider of its own, the same key and (by default) the same
    model as the main agent; `_client` is the seam a test replaces, as the desk's."""

    def __init__(self, cfg: Any, backend: Any, store: Any, home: str):
        self.backend = backend
        self.store = store
        self.home = str(home or "").rstrip("/")
        self.model = cfg.geometry_model or cfg.model
        # The desk's own effort, not the main loop's: the hosted app runs the loop at
        # medium, and at medium the model places an arc's end right one time in six.
        self.effort = cfg.geometry_effort or "high"
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
                # A refused spec still has its picture when a check failed after the
                # build (the red crosses are on it); a spec the parser refused has none.
                refused: list[dict] = []
                if png:
                    refused.append(images.attachment(images.downscale(png, "image/png"), "image/png"))
                refused.append({"type": "text", "text": (
                    f"The tool refused it:\n{report[-3000:]}\n\nReply with a corrected spec.")})
                messages.append({"role": "user", "content": refused})
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
        capped = "" if (agreed or disagrees) else ("laps" if laps >= MAX_LAPS else "time")
        if last_spec is None:
            return GeometryResult(error=f"no spec built within the {capped} allowed "
                                  f"({laps} lap(s), {seconds:.0f} s)", laps=laps,
                                  seconds=seconds, tokens=tokens, capped=capped)
        local = Path(self.store.fetch_dir()) / case
        try:
            self._write_case(mode, last_spec, study, local, last_scale)
        except Exception as exc:  # noqa: BLE001 - the report says what happened
            return GeometryResult(report=last_report, png=last_png, laps=laps, seconds=seconds,
                                  tokens=tokens, error=f"the case did not write: {exc}")
        remote = f"{self.home}/{case}" if self.home else case
        self.backend.put_tree(local, remote)
        result = GeometryResult(report=last_report, png=last_png, case_rel=remote, laps=laps,
                                seconds=seconds, tokens=tokens, agreed=agreed, disagrees=disagrees,
                                script="mesh2d.py" if mode == "2d" else "cad_gen.py",
                                scale=last_scale, capped=capped)
        self._finish(remote, result)
        return result

    # -- the finish: the OpenFOAM mesh, its check and its picture, on the instance ----

    def _finish(self, remote: str, result: GeometryResult) -> None:
        """Run Allmesh, checkMesh and the mesh render on the instance in one exec and
        carry the digest and the picture back in the result. Measured before this
        existed: the main agent spent four to six turns after "case written" finding
        out that nothing was meshed yet, that ./Allmesh had no exec bit, what the
        render tool's flags were. None of that is geometry."""
        exec_ = getattr(self.backend, "exec", None)
        get_file = getattr(self.backend, "get_file", None)
        if exec_ is None or get_file is None:
            return
        cmd = ("sh Allmesh > log.Allmesh 2>&1; rc=$?; tail -12 log.Allmesh; "
               # the case by its absolute path, not `.`: render.py names its ParaView
               # marker after the directory, and `.` gave it the name `.foam`
               f"python3 {TOOLBOX_DEST}/render.py \"$PWD\" --scene mesh --out renders > log.render 2>&1 "
               "|| tail -5 log.render; exit $rc")
        try:
            run = exec_(cmd, cwd=remote, timeout_s=FINISH_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - the words say what happened
            result.mesh_report = f"the mesh was not built on the instance: {exc}"
            return
        output = (getattr(run, "output", "") or "").strip()
        rc = getattr(run, "exit_code", 1)
        digest = ""
        try:
            log = get_file(f"{remote}/log.checkMesh", limit=400_000).decode("utf-8", "replace")
            mesh_digest = _mesh_digest()
            digest = mesh_digest.report(mesh_digest.parse(log)).strip()
        except Exception:  # noqa: BLE001 - no log means the mesh step did not get that far
            digest = ""
        try:
            result.mesh_png = get_file(f"{remote}/renders/mesh_z.png", limit=20_000_000)
        except Exception:  # noqa: BLE001
            result.mesh_png = None
        if rc == 0 and digest:
            result.meshed = True
            result.mesh_report = digest
        else:
            result.mesh_report = ("Allmesh did not finish (exit %s):\n%s" % (rc, output[-1500:])) \
                if rc != 0 else (output[-1500:] or "checkMesh wrote no log")
