#!/usr/bin/env python3
"""The acceptance runs: the prompt corpus, two arms, and everything each run left behind.

    python3 scripts/cad_accept.py desk --local          # the CAD desk, every case
    python3 scripts/cad_accept.py desk                 # the same, on the hosted image
    python3 scripts/cad_accept.py bash                 # OPENREYNOLDS_MESH_TOOL=0, the A/B
    python3 scripts/cad_accept.py old --only T7,T8     # the desk being replaced
    python3 scripts/cad_accept.py replay --only T1     # re-run a run's own script, from empty
    python3 scripts/cad_accept.py probes               # C1's probes, on the image
    python3 scripts/cad_accept.py report               # the table, from what is on disk

Every run writes its record to `docs/cad-acceptance/runs/<arm>/<T>.json` the moment it
ends, and the next invocation skips a prompt that already has one. That is not tidiness:
a run is minutes of hosted compute and real model spend, the service caps the workspace
at one instance so the whole set is serial, and a crash eleven runs in must not cost the
eleven. `--force` re-runs one anyway and records the attempt count.

**What a record holds, and why it is not just the verdict.** The finish check is the
floor and the prompt files are the ceiling: each names properties the request carries
that `check.py` cannot know -- a passage width, a loop count, a layer coverage read off
the mesh -- and "a property you did not measure is a property you did not build". So the
record keeps the desk's whole transcript, its accepted script, the per-patch STLs, the
mesh, the render and the findings JSON, and the property verdicts are entered by hand
against that evidence in `docs/cad-acceptance.md`. Nothing here decides whether a shape
is right. A harness that graded authoring would be the reviewer agent, which the plan
deliberately does not build.

**Two arms of workspace, and which one a number came from matters.** Without `--local`
this drives the hosted image, which is what `#8` asks for. With `--local` it drives
`LocalBackend` against the OpenFOAM install on this machine. The local arm exists
because the pinned image ships no Jupyter kernel and is network-sealed, so the desk
cannot run a single cell there without wheels carried in by hand -- a blocking item for
whoever owns the image, and not something a harness should quietly route around. A
local pass measures the desk; it does not certify the image, and every number it
produces is labelled that way.

**Geometry never executes here.** The fixtures are uploaded and the artifacts are pulled
down; every triangle is read on the instance by code that runs there.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPTS = ROOT / "tests" / "data" / "prompts"
RESULTS = ROOT / "docs" / "cad-acceptance"
FIXTURES = PROMPTS / "fixtures"

ARMS = ("desk", "bash", "old")

WORKSPACE_ROOT = "/work"
"""What the protocol advertises as the workspace root, on either backend."""

REMOTE_FIXTURES = f"{WORKSPACE_ROOT}/.accept"
"""Where the two STEP fixtures live on the instance.

Under the workspace root because `#12`'s `geometry` property is checked for existence
and readability before a run starts, and a path outside the root is not a path the tool
will take."""

WHEELS_LOCAL = Path(
    os.environ.get("OPENREYNOLDS_ACCEPT_WHEELS", "")) if os.environ.get(
    "OPENREYNOLDS_ACCEPT_WHEELS") else None
"""A local directory of wheels to carry onto the image (see `ensure_kernel_deps`).

Not committed: fifteen megabytes of binary wheels in a source tree is the wrong fix
for a missing image dependency, and the right one is the image."""

from openreynolds.llm.presets import PRICE_PER_MTOK, prices  # noqa: E402
from openreynolds.llm.presets import spend as presets_spend  # noqa: E402
"""Priced by model, from the package where the model ids already live.

This was a single untagged table here, holding Sonnet 5's rates because Sonnet 5 is the
default preset -- while the build-up sweep runs Opus 5. Every dollar the first baseline
reported was 2.5x under."""


# -- the prompts ---------------------------------------------------------------


def load_prompts() -> dict[str, dict[str, Any]]:
    """The corpus, read off the committed files rather than restated here.

    Whatever `T*.md` is on disk is the corpus -- it was eight and is twelve, and the
    count lives nowhere but the directory, so adding a case is adding a file.

    The request is the blockquote under `## Request`; the named properties are the
    bullets under the properties heading, which is what each run is judged on beyond
    the finish check. T5 and T6 name a fixture, which is what `geometry` carries.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(PROMPTS.glob("T*.md")):
        name = path.name.split("-")[0]
        text = path.read_text(encoding="utf-8")
        out[name] = {
            "name": name,
            "file": str(path.relative_to(ROOT)),
            "title": text.splitlines()[0].lstrip("# ").strip(),
            "request": _blockquote(text, "## Request"),
            "properties": _bullets(text, "## Properties the desk must measure and print"),
            "false_pass": _section(text, "## A pass that is really a failure"),
            "expects": _expects(text),
            "geometry": _geometry_for(name),
        }
    return out


_EXPECTS = re.compile(r"^\*\*Passes as:\*\*\s*`([a-z-]+)`", re.M)


def _expects(text: str) -> str:
    """The terminal state this case counts as a pass. `done` unless it says otherwise.

    Read off the prompt rather than kept in a table here, for the same reason the request
    and the properties are: the case's definition is the case's file. Only T6 says
    anything but `done`, and T6 is the reason this exists -- its pass is the desk
    declining, and the harness had no way to know that, so it scored the guess as the
    corpus's sixth success."""
    found = _EXPECTS.search(text or "")
    return found.group(1) if found else "done"


def _geometry_for(name: str) -> str:
    """The remote path of the fixture a prompt hands the desk, or ""."""
    if name == "T5":
        return f"{REMOTE_FIXTURES}/assembly.step"
    if name == "T6":
        return f"{REMOTE_FIXTURES}/customer_part.step"
    return ""


LOCAL_FIXTURES = {
    "assembly.step": ROOT / "tests" / "data" / "cad" / "real" / "ldrobot_ld19_lidar.step",
    "customer_part.step": FIXTURES / "no_unit_box_with_duct.step",
}
"""T5 gets C1's real-CAD multi-solid file. T6 gets the same synthetic box-with-a-duct
C1 built, with its `LENGTH_UNIT` declaration emptied -- the one property T6 is about,
and the only edit. Named `customer_part.step` on the instance because the request says
it came from a customer and a filename is a hint the desk should not be given."""


def _blockquote(text: str, heading: str) -> str:
    body = _section(text, heading)
    lines = [line[1:].strip() for line in body.splitlines() if line.startswith(">")]
    return " ".join(line for line in lines if line).strip()


def _bullets(text: str, heading: str) -> list[str]:
    out: list[str] = []
    for line in _section(text, heading).splitlines():
        if line.startswith("- "):
            out.append(line[2:].strip())
        elif out and line.startswith("  ") and line.strip():
            out[-1] += " " + line.strip()
    return [re.sub(r"\s+", " ", b) for b in out]


def _section(text: str, heading: str) -> str:
    lines = text.splitlines()
    try:
        start = lines.index(heading) + 1
    except ValueError:
        return ""
    end = start
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    return "\n".join(lines[start:end]).strip()


# -- where a run is kept -------------------------------------------------------


def record_path(arm: str, name: str) -> Path:
    return RESULTS / "runs" / arm / f"{name}.json"


def artifact_dir(arm: str, name: str) -> Path:
    return RESULTS / "artifacts" / arm / name


def load_record(arm: str, name: str) -> dict[str, Any] | None:
    path = record_path(arm, name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_record(arm: str, name: str, record: dict[str, Any]) -> None:
    """Write the record, now, before anything else is attempted.

    Checkpointing after every run is the whole resumability story: the file on disk is
    what a re-invocation reads, so a run that completed is never repeated and a crash
    costs the run it was in and nothing before it.
    """
    path = record_path(arm, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def spend(tokens: dict[str, int], model: str = "", provider: str = "") -> float:
    """What these tokens cost at this `model`'s rates at this `provider`.

    `model` is defaulted rather than required only so an old call site is a wrong number
    instead of a crash while it is being updated -- and an empty model prices at zero,
    which is loud. Every call site in the tree passes one.

    `provider` is what makes the rate right where a model is served by more than one
    vendor: kimi-k3 is 20% dearer at Moonshot than at Aster, and a table keyed on the
    model id alone cannot say which run it is pricing."""
    return presets_spend(tokens, model, provider)


# -- the workspace -------------------------------------------------------------


def impatient_stat(backend: Any) -> Any:
    """Make this backend's `stat` give up at once on a file that is not there.

    **A workaround for a defect in the transport, recorded here because every number
    in this report depends on it.** The kernel channel's protocol is to poll
    `stat(done-N.json)` until the cell writes it, so the overwhelmingly common case is
    statting a file that is legitimately not there yet. The hosted service answers a
    missing file with **503 `sandbox_unavailable`** carrying "(No such file or
    directory)" in its message; 503 is in `hosted.py`'s retry set, so the client tries
    five times with backoff. Measured against the live service:

        stat, file present   1.9 - 2.8 s
        stat, file absent   31.9 - 34.6 s     <- five attempts at a 404 in 503 clothing
        stat, absent, one attempt  2.0 s
        get_file  2.1 - 2.4 s   put_file  2.6 - 3.7 s   exec  3.2 - 3.7 s

    The arithmetic that follows: `1+1` through `kernel_run` took **79.5 s** wall clock
    while the kernel itself reported **0.01 s**, and a whole cell costs 80-105 s
    whatever is in it. At the desk's 900 s budget that is nine cells a run, which is
    why T7 -- "short; the shape is simple" -- spent its entire budget and never reached
    a mesher. It is not the desk being slow; it is the desk waiting on retries against
    a file whose absence is the protocol working.

    Capping `stat` at one attempt is safe *in this protocol* and nowhere else: a false
    "not there" costs one more lap of a loop that is already looping. It is bound onto
    the instance rather than wrapped around it because the kernel channel holds the
    backend itself (`KernelHost._channel` is built with `self`), so a proxy in front
    of it is not on the path that matters. It is done here, in the harness, rather
    than in `backend/kernel.py`, because that file belongs to another chunk -- the
    defect is reported, not patched.
    """
    inner = type(backend).stat

    def stat(path: str, *, timeout: float = 300.0, max_attempts: int | None = None):
        return inner(backend, path, timeout=timeout, max_attempts=max_attempts or 1)

    backend.stat = stat
    return backend


class _NoClient:
    """Stands in for the hosted client in local mode, so `finally: client.close()` holds.

    The hosted arm owns an HTTP session and an instance that has to be released; the
    local arm owns a directory. Giving the local arm the same shape means the drivers
    below do not each grow a branch for which kind of workspace they are on."""

    def close(self) -> None:
        return None


LOCAL_WORK = ROOT / ".accept-work"
"""The fallback local workspace, when `/work` itself cannot be used.

Inside the tree but ignored, because everything a run writes -- the case directories,
the kernel sockets, the replay scratch -- is output, and the artifact set pulled into
`docs/cad-acceptance/artifacts/` is what is committed."""


def workspace_root_for(work: Path | None) -> tuple[Path, str]:
    """The local root, and a warning if it is not the one the old desk still assumes.

    **The new desk no longer cares, and that is a fix this chunk had to make.**
    `cad/brief.py` told the desk its instruments "are in `/work/.toolbox/`" as literal
    text and `cad/check.py` ran `python3 /work/.toolbox/mesh_look.py` through a shell.
    Both are true on the hosted image and false on `LocalBackend`, which is rooted
    wherever it was told to be -- and nothing translates `/work` for a shell or for a
    `subprocess.run` inside a cell. The measured cost, in the first local T1 run: seven
    of twenty-seven steps spent on `grep`, `ls`, `find /` and `ls ../.toolbox` before
    the desk found the tools it had just been handed, and a finish check that reported
    "the check wrote no readable answer" about a mesh it never opened. Both now take
    the path from `backend.workspace_root`, pinned from both ends in
    `tests/test_cad_brief_matches_the_cell_log.py`.

    **The old desk still assumes it**, and deliberately is not fixed: `mesher/brief.py`
    and `mesher/check.py` are the code being replaced, and T7 and T8 are run against
    them exactly as they are. So the local arm is rooted at `/work` anyway, where both
    halves are right, and the fallback below is reported rather than taken quietly.
    """
    if work:
        chosen = work.expanduser().resolve()
        chosen.mkdir(parents=True, exist_ok=True)
        warn = "" if str(chosen) == "/work" else (
            f"workspace is {chosen}, not /work: the brief and cad/check.py name "
            "/work/.toolbox literally, so the desk will be misdirected and the finish "
            "check will not run its own tools")
        return chosen, warn
    candidate = Path("/work")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".writable"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return candidate.resolve(), ""
    except OSError as exc:
        LOCAL_WORK.mkdir(parents=True, exist_ok=True)
        return LOCAL_WORK.resolve(), (
            f"/work is not usable ({exc}); falling back to {LOCAL_WORK}. Every number "
            "from this run is affected: see workspace_root_for.")


def open_local(work: Path | None = None):
    """The same five things, against OpenFOAM on this machine.

    **Why this exists at all is a finding, and it is recorded rather than papered
    over.** The hosted path below is the one the plan frames the acceptance around --
    "the desk on the image" -- and it is left exactly as it was. But the pinned image
    ships no Jupyter kernel (see `ensure_kernel_deps`) and is network-sealed, so
    nothing on it can run a cell without wheels carried in from outside. Running here
    measures the desk, honestly and end to end, on a machine that has OpenFOAM v2512,
    cfMesh, gmsh, build123d and a kernel already; it does not measure the image, and
    `docs/cad-acceptance.md` says so in its first paragraph.

    `LocalBackend` accepts `/work` as an alias for its real root, so the prompts, the
    toolbox destination and the fixture paths are the same strings in both arms."""
    from openreynolds.backend.local import LocalBackend, find_bashrc
    from openreynolds.config import Config
    from openreynolds.store import Store, new_study_id

    cfg = Config.load()
    key = cfg.model_key_missing()
    if key:
        # Only the model key is required here. `FOAMD_URL`/`FOAMD_API_KEY` gate the
        # hosted workspace, and there isn't one.
        raise SystemExit(f"missing configuration: {key}")
    bashrc = find_bashrc()
    if not bashrc:
        raise SystemExit("no OpenFOAM installation found; set OPENREYNOLDS_FOAM_BASHRC")
    root, warn = workspace_root_for(work)
    if warn:
        print(f"  WARNING: {warn}")
    backend = LocalBackend(root=root, bashrc=bashrc)
    store = Store(cfg.studies_dir, "accept-" + new_study_id())
    return cfg, backend, _NoClient(), f"local:{root}", store


def open_workspace(instance: str | None = None, patient: bool = True,
                   local: bool = False, work: Path | None = None):
    """A config, a backend and a study directory to write the transcript into.

    `local=True` takes the branch above; everything else on this path is the hosted
    arm unchanged.
    """
    if local:
        return open_local(work)
    from openreynolds.backend import hosted
    from openreynolds.config import Config
    from openreynolds.store import Store, new_study_id

    cfg = Config.load()
    missing = cfg.missing()
    if missing:
        raise SystemExit(f"missing configuration: {', '.join(missing)}")
    backend, client, instance_id = hosted.acquire(cfg.foamd_url, cfg.foamd_api_key, instance)
    store = Store(cfg.studies_dir, "accept-" + new_study_id())
    if patient:
        impatient_stat(backend)
    return cfg, backend, client, instance_id, store


def sync_toolbox(backend) -> None:
    """The toolbox, where `cli.py` puts it. `check.py` reaches for it by that path."""
    backend.put_tree(ROOT / "openreynolds" / "toolbox", "/work/.toolbox")


KERNEL_DEPS = ("jupyter_client", "ipykernel", "pyzmq")
PYDEPS = "/work/.pydeps"
WHEELS = "/work/.wheels"

_ENSURE = f"""
python3 -c 'import jupyter_client, ipykernel, zmq' 2>/dev/null && {{ echo already; exit 0; }}
[ -d {PYDEPS} ] || python3 -m pip install --no-index --find-links {WHEELS} \
  --target {PYDEPS} --upgrade {' '.join(KERNEL_DEPS)} 2>&1 | tail -2
SP=$(python3 -c 'import site; print(site.getsitepackages()[0])')
echo {PYDEPS} > "$SP/openreynolds_accept.pth"
python3 -c 'import jupyter_client, ipykernel, zmq; print("side-loaded", jupyter_client.__version__, ipykernel.__version__, zmq.__version__)'
"""


def ensure_kernel_deps(backend, wheels: Path | None = None) -> str:
    """Put a Jupyter kernel on the image, because the image does not ship one.

    **This is a finding, not a convenience.** C8a's kernel channel runs the desk's
    cells through `jupyter_client.manager.KernelManager`, and the pinned image carries
    IPython and traitlets but not `jupyter_client`, `ipykernel`, `pyzmq`, `tornado`,
    `jupyter_core`, `comm` or `debugpy`. The image is also network-sealed -- `pip
    install` there answers "needs the network, which is sealed in this sandbox" -- so
    the dependency cannot be satisfied at run time by the thing that needs it. Without
    this the desk returns `no kernel on this workspace: No module named
    'jupyter_client'` and every one of the prompts fails before a model is
    called, which is how it was found.

    So the wheels are carried in from the dev machine, unpacked onto the persistent
    volume, and put on `sys.path` **after** site-packages by a `.pth` file, so
    everything the image already has still wins. It is reproducible and it is recorded
    in `docs/cad-acceptance.md`, and it is not a fix: the image is what needs changing,
    and until it is, nothing on it can run a cell.
    """
    if wheels and wheels.is_dir():
        backend.put_tree(wheels, WHEELS)
    outcome = backend.exec(_ENSURE, timeout_s=900)
    return (outcome.output or "").strip()[-400:]


def push_fixtures(backend) -> dict[str, str]:
    """Put the two STEP files on the instance and say what they are.

    `declared_unit` is read here on the local copy only -- it is a text scan of the
    file, not a kernel call -- so the record can state that T6's fixture really does
    declare nothing before any model is asked about it.
    """
    sys.path.insert(0, str(ROOT / "openreynolds" / "toolbox"))
    import cad_convert  # noqa: E402  (sibling script, not a package)

    backend.exec(f"mkdir -p {shlex.quote(REMOTE_FIXTURES)}", timeout_s=60)
    facts: dict[str, str] = {}
    for name, local in LOCAL_FIXTURES.items():
        backend.put_file(f"{REMOTE_FIXTURES}/{name}", local.read_bytes())
        facts[name] = json.dumps(cad_convert.declared_unit(local))
    return facts


# -- collecting what a run left behind -----------------------------------------

_LIST = r"""
set -e
cd %s 2>/dev/null || exit 0
find . -maxdepth 6 \( -name '*.stl' -o -name '*.png' -o -name 'patches.json' \
  -o -name 'build.py' -o -name 'boundary' -o -name '*.json' \) -type f 2>/dev/null | head -200
"""


def collect(backend, case_dir: str, out: Path, extra: dict[str, bytes] | None = None) -> list[str]:
    """Pull the artifact set down: the script, the per-patch STLs, the render, the JSON.

    Best effort by design. A run that failed halfway has half a case directory, and
    half of it is still the evidence for what went wrong.
    """
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (extra or {}).items():
        (out / name).write_bytes(data)
    try:
        listing = backend.exec(_LIST % shlex.quote(case_dir), timeout_s=120)
    except Exception as exc:  # noqa: BLE001 - the workspace, not the mesh
        (out / "collect-error.txt").write_text(str(exc), encoding="utf-8")
        return []
    names = [line.strip().lstrip("./") for line in (listing.output or "").splitlines()
             if line.strip().startswith("./")]
    got: list[str] = []
    for rel in names:
        if "/replay/" in rel:
            continue
        try:
            data = backend.get_file(f"{case_dir.rstrip('/')}/{rel}")
        except Exception:  # noqa: BLE001 - one missing file is not a failed collection
            continue
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        got.append(rel)
    # The mesh itself, as numbers rather than as a polyMesh: pulling several hundred
    # megabytes of `constant/polyMesh` per run would be the artifact set nobody can
    # commit. The boundary file and checkMesh's own words are what a reader needs.
    try:
        summary = backend.exec(
            "for d in constant/polyMesh constant/*/polyMesh; do "
            "[ -d \"$d\" ] || continue; echo \"== $d\"; "
            "wc -l \"$d/owner\" 2>/dev/null | head -1; "
            "head -30 \"$d/boundary\" 2>/dev/null; done", cwd=case_dir, timeout_s=120)
        (out / "polymesh-summary.txt").write_text(summary.output or "", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return got


# -- the desk arm --------------------------------------------------------------


def run_desk(cfg, backend, store, prompt: dict[str, Any], out: Path) -> dict[str, Any]:
    """One prompt against the CAD desk, on the instance, for real."""
    from openreynolds import cad

    desk = cad.CadDesk(cfg, backend, store, "/work")
    started = time.time()
    result = desk.run(prompt["request"], case=f"accept-{prompt['name'].lower()}",
                      geometry=prompt["geometry"])
    wall = time.time() - started

    extra: dict[str, bytes] = {}
    if result.script:
        # The log is the deliverable, and it has to be on disk on both sides: in the
        # artifact set for a reader, and in the case directory so `replay` can run it
        # where the geometry is.
        extra["build.py"] = result.script.encode("utf-8")
        try:
            backend.put_file(f"{result.case_dir}/build.py", result.script.encode("utf-8"))
        except Exception:  # noqa: BLE001
            pass
    if result.png:
        extra["render.png"] = result.png
    if result.check is not None:
        extra["check.json"] = json.dumps(
            {"ok": result.check.ok, "unreachable": result.check.unreachable,
             "missing": result.check.missing, "regions": result.check.regions,
             "cells": result.check.cells, "bounds": result.check.bounds,
             "patches": result.check.patches, "checkmesh": result.check.checkmesh,
             "metrics": result.check.metrics, "two_d": result.check.two_d,
             "build": result.check.build, "error": result.check.error,
             "findings": [f._asdict() for f in result.check.findings],
             "lines": result.check.lines(), "raw": result.check.raw},
            indent=2, default=str).encode("utf-8")
    extra["steps.json"] = json.dumps(
        [asdict(s) for s in result.steps], indent=2, default=str).encode("utf-8")

    files = collect(backend, result.case_dir, out, extra)
    return {
        "ok": bool(result.ok),
        "stopped": result.stopped,
        "error": result.error,
        "summary": result.summary,
        "case_dir": result.case_dir,
        "seconds": round(wall, 1),
        "desk_seconds": round(result.seconds, 1),
        "steps": len(result.steps),
        "tokens": dict(result.tokens),
        "usd": round(spend(result.tokens, cfg.mesher_model or cfg.model, cfg.provider), 4),
        "script_lines": len((result.script or "").splitlines()),
        "check": _check_digest(result.check),
        "artifacts": sorted(files) + sorted(extra),
    }


def _check_digest(check) -> dict[str, Any]:
    if check is None:
        return {"ran": False}
    return {
        "ran": True, "ok": bool(check.ok), "unreachable": bool(check.unreachable),
        "missing": list(check.missing), "regions": list(check.regions),
        "cells": check.cells, "two_d": bool(check.two_d),
        "patches": [p.get("name") for p in (check.patches or [])],
        "findings": [{"check": f.check, "status": f.status, "measured": f.measured}
                     for f in (check.findings or [])],
        "replay": _replay_digest(check.raw),
    }


def _replay_digest(raw: Any) -> dict[str, Any]:
    replay = (raw or {}).get("replay") if isinstance(raw, dict) else None
    if not isinstance(replay, dict) or not replay.get("ran"):
        return {"ran": False}
    return {"ran": True, "exit_code": replay.get("exit_code"),
            "before": replay.get("before"), "after": replay.get("after"),
            "output_tail": (replay.get("output") or "")[-1200:]}


# -- the old desk, for T7 and T8 -----------------------------------------------


def run_old(cfg, backend, store, prompt: dict[str, Any], out: Path) -> dict[str, Any]:
    """The same request handed to the desk being replaced.

    Not a fair fight and not meant to be one: T7 and T8 are the two cases §5 says it
    cannot return at all, and the point of running them is that the refusal is on the
    record from the thing itself rather than inferred from reading its source.
    """
    from openreynolds import mesher

    desk = mesher.Mesher(cfg, backend, store, "/work")
    started = time.time()
    result = desk.run(prompt["request"], case=f"old-{prompt['name'].lower()}")
    wall = time.time() - started
    extra: dict[str, bytes] = {}
    # The old arm's transcript, kept for the same reason the new one's is: T7 and T8
    # exist to show what the old desk does with them, and "it refused" is a claim until
    # the refusal is on disk in its own words.
    extra["steps.json"] = json.dumps(
        [asdict(s) for s in result.steps], indent=2, default=str).encode("utf-8")
    if result.png:
        extra["render.png"] = result.png
    if result.check is not None:
        extra["check.json"] = json.dumps(
            {"ok": result.check.ok, "missing": result.check.missing,
             "cells": getattr(result.check, "cells", 0),
             "patches": getattr(result.check, "patches", []),
             "lines": result.check.lines(), "raw": result.check.raw},
            indent=2, default=str).encode("utf-8")
    files = collect(backend, result.case_dir, out, extra)
    return {
        "ok": bool(result.ok), "stopped": result.stopped, "error": result.error,
        "summary": result.summary, "case_dir": result.case_dir,
        "seconds": round(wall, 1), "steps": len(result.steps),
        "tokens": dict(result.tokens), "usd": round(spend(result.tokens, cfg.mesher_model or cfg.model, cfg.provider), 4),
        "missing": list(result.check.missing) if result.check else [],
        "artifacts": sorted(files) + sorted(extra),
    }


def old_check_on(backend, case_dir: str, case_rel: str, request: str) -> dict[str, Any]:
    """The old finish check, pointed at a case the new desk built.

    The tighter half of the before/after. An end-to-end old-desk run changes two things
    at once -- a different brief and a different check -- so a refusal from it does not
    on its own say which one refused. Running `mesher/check.py` over the *new* desk's
    accepted case holds the geometry fixed and leaves only the rule.
    """
    from openreynolds.mesher import check as oldcheck

    verdict = oldcheck.verify(backend, case_dir, case_rel, request)
    return {"ok": bool(verdict.ok), "missing": list(verdict.missing),
            "cells": getattr(verdict, "cells", 0),
            "patches": [p.get("name") for p in (getattr(verdict, "patches", None) or [])],
            "lines": verdict.lines()}


# -- the bash arm --------------------------------------------------------------


def run_bash(cfg, prompt: dict[str, Any], out: Path, instance: str,
             local: bool = False, work: str | None = None) -> dict[str, Any]:
    """The same request to the main agent with the `cad` tool taken away.

    `OPENREYNOLDS_MESH_TOOL=0` is the toggle `5b588c3` added to make exactly this
    measurable, and it is kept working as an alias for that reason. A subprocess rather
    than an in-process loop because what is being compared is the product as a caller
    gets it -- the whole session, its toolbox sync, its own accounting -- not a loop
    with the tool list edited.
    """
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OPENREYNOLDS_MESH_TOOL"] = "0"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if local:
        # `OPENREYNOLDS_LOCAL` is the product's own switch (`cli.py`), not one added
        # for the harness: the arm stays "the product as a caller gets it", pointed at
        # the same workspace directory the desk arm used so both see one toolbox and
        # one set of fixtures.
        env["OPENREYNOLDS_LOCAL"] = "1"
        if work:
            env["OPENREYNOLDS_LOCAL_WORK"] = str(work)
    ask = (prompt["request"] + "\n\nBuild the geometry and mesh it in a case directory "
           f"called `bash-{prompt['name'].lower()}` under the study directory.")
    if prompt["geometry"]:
        ask += f"\n\nThe CAD file is at {prompt['geometry']} on the workspace."
    argv = [sys.executable, "-m", "openreynolds", "-p", ask, "--no-capture", "--plain"]
    if not local:
        argv += ["--instance", instance, "--keep-alive", "--max-wait", "20"]
    started = time.time()
    proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=3600)
    wall = time.time() - started
    (out / "session.log").write_text((proc.stdout or "") + "\n" + (proc.stderr or ""),
                                     encoding="utf-8")
    text = (proc.stdout or "") + (proc.stderr or "")
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "seconds": round(wall, 1),
        "steps": _steps_from_log(text),
        "tokens": _tokens_from_log(text),
        "study": _study_from_log(text),
        "tail": (proc.stdout or "")[-3000:],
    }


_STEPS = re.compile(r"^\s*--\s*step\s+(\d+):", re.M)


def _steps_from_log(text: str) -> int:
    """How many turns the session took, off the step banner it prints for a person.

    The only measure of effort this arm reports at all -- see `_tokens_from_log`."""
    found = _STEPS.findall(text)
    return max((int(n) for n in found), default=0)


_TOKENS = re.compile(r"(input|output|cache_read|cache_write)[^0-9]{0,4}([\d,]+)")


def _tokens_from_log(text: str) -> dict[str, int]:
    """What the session said it spent, off its own accounting line -- if it said.

    **Measured: it does not say, and that is a finding.** `Loop.token_totals` is kept
    and split four ways precisely so the cache share can be seen, and nothing writes
    it anywhere a reader or a parser can reach: it is not in `session.json`, not in
    `messages.jsonl`, and not printed by `-p --plain`. Only the interactive `/status`
    view shows a single `context_tokens` number, which is one number for four prices
    that span 250x.

    So this returns `{}` for a plain-mode session, and the A/B's cost half is reported
    as not recoverable from the product's own output rather than estimated. Inventing a
    second accounting path inside the product to make this arm tidier would be editing
    the thing under test; guessing from character counts would be worse, because the
    bill here is dominated by cache reads of a growing prefix and a character count
    cannot see them.
    """
    found: dict[str, int] = {}
    for key, value in _TOKENS.findall(text[-8000:]):
        found[key] = max(found.get(key, 0), int(value.replace(",", "")))
    return found


def _study_from_log(text: str) -> str:
    match = re.search(r"\b(\d{8}-\d{6}-[0-9a-f]{4})\b", text)
    return match.group(1) if match else ""


# -- the replay criterion ------------------------------------------------------


def run_replay(backend, prompt: dict[str, Any], record: dict[str, Any],
               out: Path) -> dict[str, Any]:
    """Run the run's own script in an empty directory, and check the mesh it makes.

    `check.verify` already replays the script and fingerprints the geometry; what it
    does not do is put the finish check back over the result, because at that point in
    a run the mesh is the one being judged. §3's invariant is the stronger claim -- the
    artifact reproduces a mesh that *passes* -- so it is measured here, on real output.
    """
    from openreynolds import cad

    script = (artifact_dir("desk", prompt["name"]) / "build.py")
    if not script.exists():
        return {"ran": False, "why": "no accepted script was left behind"}
    case_dir = f"/work/replay-{prompt['name'].lower()}"
    backend.exec(f"rm -rf {shlex.quote(case_dir)} && mkdir -p {shlex.quote(case_dir)}",
                 timeout_s=120)
    backend.put_file(f"{case_dir}/build.py", script.read_bytes())
    started = time.time()
    outcome = backend.exec("python3 build.py > replay.log 2>&1; echo exit=$?; tail -60 replay.log",
                           cwd=case_dir, timeout_s=2400)
    ran = time.time() - started
    verdict = cad.verify(backend, case_dir, f"replay-{prompt['name'].lower()}",
                         prompt["request"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay.log").write_text(outcome.output or "", encoding="utf-8")
    return {"ran": True, "seconds": round(ran, 1),
            "script_exit": outcome.output.strip().split("exit=")[-1].splitlines()[0]
            if "exit=" in (outcome.output or "") else "?",
            "check": _check_digest(verdict),
            "output_tail": (outcome.output or "")[-2000:]}


# -- the probes, on the image --------------------------------------------------


def run_probes(backend, out: Path) -> dict[str, Any]:
    """C1's probes, where they were always owed: on the image.

    C1 measured them on the dev machine's OCP 7.9.3.1.1 against the image's 7.8.1 and
    said so -- `versions` is compared like any other answer, so the run is expected to
    go red on the OCC version before it says anything about geometry. What is recorded
    here is the whole `--dump`, so which answers moved between the two versions and
    which did not is a diff rather than a claim.
    """
    out.mkdir(parents=True, exist_ok=True)
    backend.exec("mkdir -p /work/.probes", timeout_s=60)
    for rel in ("scripts/cad_probes.py",):
        backend.put_file(f"/work/.probes/{Path(rel).name}", (ROOT / rel).read_bytes())
    # The probes import the package and read the committed fixtures, so both go up.
    backend.put_tree(ROOT / "tests" / "data" / "cad", "/work/.probes/tests/data/cad")
    backend.put_tree(ROOT / "openreynolds" / "toolbox", "/work/.probes/openreynolds/toolbox")
    backend.put_file("/work/.probes/openreynolds/__init__.py", b"")
    results: dict[str, Any] = {}
    for label, cmd in (
        ("compare", "python3 scripts/cad_probes.py --no-pytest; echo exit=$?"),
        ("dump", "python3 scripts/cad_probes.py --dump --no-pytest"),
    ):
        try:
            outcome = backend.exec(cmd, cwd="/work/.probes", timeout_s=1800)
            results[label] = outcome.output or ""
        except Exception as exc:  # noqa: BLE001
            results[label] = f"unavailable: {exc}"
        (out / f"probes-{label}.txt").write_text(results[label], encoding="utf-8")
    return results


# -- driving ------------------------------------------------------------------


def drive(arm: str, only: list[str], force: bool, instance: str | None,
          local: bool = False, work: Path | None = None) -> int:
    prompts = load_prompts()
    chosen = [p for name, p in prompts.items() if not only or name in only]
    if not chosen:
        print(f"no prompts matched {only}")
        return 2

    cfg, backend, client, instance_id, store = open_workspace(instance, local=local, work=work)
    print(f"instance {instance_id}")
    try:
        sync_toolbox(backend)
        print("  kernel: " + ensure_kernel_deps(backend, WHEELS_LOCAL))
        facts = push_fixtures(backend)
        for name, evidence in facts.items():
            print(f"  fixture {name}: {evidence}")

        for prompt in chosen:
            name = prompt["name"]
            existing = load_record(arm, name)
            if existing and existing.get("completed") and not force:
                print(f"{arm}/{name}: already run ({existing.get('verdict', '?')}), skipping")
                continue
            attempts = int((existing or {}).get("attempts", 0)) + 1
            out = artifact_dir(arm, name)
            record: dict[str, Any] = {
                "arm": arm, "prompt": name, "title": prompt["title"],
                "max_seconds": float(cfg.mesher_max_seconds or 0),
                "max_steps": int(cfg.mesher_max_steps or 0),
                "request": prompt["request"], "properties": prompt["properties"],
                "geometry": prompt["geometry"], "attempts": attempts,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "completed": False, "model": cfg.mesher_model or cfg.model,
                "effort": cfg.mesher_effort or cfg.effort,
            }
            if existing:
                # A retried run keeps the one it replaces. "Do not retry until green
                # and report only the green one" is the rule; carrying the earlier
                # attempt in the same file is what makes following it checkable.
                record["previous"] = [
                    *(existing.get("previous") or []),
                    {k: v for k, v in existing.items() if k != "previous"},
                ]
            save_record(arm, name, record)
            print(f"{arm}/{name}: {prompt['title']} (attempt {attempts}) ...", flush=True)
            started = time.time()
            try:
                if arm == "desk":
                    record["run"] = run_desk(cfg, backend, store, prompt, out)
                elif arm == "old":
                    record["run"] = run_old(cfg, backend, store, prompt, out)
                elif arm == "bash":
                    record["run"] = run_bash(cfg, prompt, out, instance_id,
                                             local=local, work=backend.workspace_root)
                else:
                    raise SystemExit(f"unknown arm {arm}")
                record["completed"] = True
            except BaseException as exc:  # noqa: BLE001 - a crash is a result too
                record["crashed"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()[-4000:]
                record["completed"] = False
                save_record(arm, name, record)
                print(f"  crashed: {record['crashed']}")
                if isinstance(exc, KeyboardInterrupt):
                    raise
                continue
            record["seconds"] = round(time.time() - started, 1)
            record["verdict"] = "check-ok" if record["run"].get("ok") else "check-not-ok"
            save_record(arm, name, record)
            run = record["run"]
            print(f"  {record['verdict']} in {record['seconds']}s, "
                  f"${run.get('usd', 0):.2f}, stopped={run.get('stopped', '')!r}")
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


def drive_replay(only: list[str], force: bool, instance: str | None,
                 local: bool = False, work: Path | None = None) -> int:
    prompts = load_prompts()
    cfg, backend, client, instance_id, store = open_workspace(instance, local=local, work=work)
    print(f"instance {instance_id}")
    try:
        sync_toolbox(backend)
        ensure_kernel_deps(backend, WHEELS_LOCAL)
        for name, prompt in prompts.items():
            if only and name not in only:
                continue
            desk = load_record("desk", name)
            if not desk or not desk.get("completed"):
                continue
            existing = load_record("replay", name)
            if existing and existing.get("completed") and not force:
                print(f"replay/{name}: already run, skipping")
                continue
            record = {"arm": "replay", "prompt": name, "completed": False,
                      "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            save_record("replay", name, record)
            print(f"replay/{name} ...", flush=True)
            try:
                record["run"] = run_replay(backend, prompt, desk,
                                           artifact_dir("replay", name))
                record["completed"] = True
            except BaseException as exc:  # noqa: BLE001
                record["crashed"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()[-4000:]
                save_record("replay", name, record)
                if isinstance(exc, KeyboardInterrupt):
                    raise
                continue
            save_record("replay", name, record)
            print(f"  {json.dumps(record['run'].get('check', {}).get('ok'))}")
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


def drive_old_check(only: list[str], instance: str | None,
                    local: bool = False, work: Path | None = None) -> int:
    """The old finish check over the new desk's own accepted cases."""
    prompts = load_prompts()
    cfg, backend, client, instance_id, store = open_workspace(instance, local=local, work=work)
    try:
        sync_toolbox(backend)
        for name, prompt in prompts.items():
            if only and name not in only:
                continue
            desk = load_record("desk", name)
            if not desk or not desk.get("completed"):
                continue
            case_dir = desk["run"].get("case_dir", "")
            record = {"arm": "oldcheck", "prompt": name, "case_dir": case_dir,
                      "completed": False}
            save_record("oldcheck", name, record)
            print(f"oldcheck/{name} on {case_dir} ...", flush=True)
            record["run"] = old_check_on(backend, case_dir,
                                         f"accept-{name.lower()}", prompt["request"])
            record["completed"] = True
            save_record("oldcheck", name, record)
            print(f"  old check ok={record['run']['ok']} missing={record['run']['missing']}")
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


def drive_bash_check(only: list[str], instance: str | None,
                     local: bool = False, work: Path | None = None) -> int:
    """The desk's own finish check, run over what the bash arm left behind.

    **Without this the A/B compares two different things.** The bash arm's record says
    only whether the process exited 0, which every session does whether it built a mesh
    or explained why it could not; the desk arm's says whether the mesh passed `#3`'s
    check. So the same check is run over the bash arm's case directory afterwards, and
    the two columns in `docs/cad-acceptance.md` are then the same measurement.

    It is run after the fact rather than inside the session because the point of the
    arm is the product as a caller gets it, and a caller does not get this check.
    """
    from openreynolds import cad

    prompts = load_prompts()
    cfg, backend, client, instance_id, store = open_workspace(instance, local=local, work=work)
    try:
        sync_toolbox(backend)
        for name, prompt in prompts.items():
            if only and name not in only:
                continue
            rec = load_record("bash", name)
            if not rec or not rec.get("completed"):
                continue
            study = rec["run"].get("study", "")
            case_rel = f"bash-{name.lower()}"
            case_dir = f"{WORKSPACE_ROOT}/{study}/{case_rel}" if study else ""
            if not case_dir:
                rec["verify"] = {"ran": False, "why": "the session log named no study"}
                save_record("bash", name, rec)
                continue
            print(f"bashcheck/{name} on {case_dir} ...", flush=True)
            verdict = cad.verify(backend, case_dir, case_rel, prompt["request"])
            rec["verify"] = {"ran": True, "case_dir": case_dir,
                             "check": _check_digest(verdict)}
            save_record("bash", name, rec)
            print(f"  ok={verdict.ok} cells={verdict.cells} missing={len(verdict.missing)}")
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


def drive_recheck(only: list[str], instance: str | None,
                  local: bool = False, work: Path | None = None) -> int:
    """The finish check again, with the accepted script on disk where I5 says it is.

    **Why this exists, and it is a finding rather than a convenience.** I5 and the brief
    both promise the desk that "your accepted cells are the script you leave behind ...
    concatenated into `build.py` in the case directory". Nothing writes it. The only
    place the concatenation is ever materialised is the replay directory `check._replay`
    makes and then deletes, so at the moment the finish check reads the case there is no
    `build.py` in it, and `Check.build` fails with "there is no Allmesh (or build script)
    in the case ... leave the script that made this mesh" -- about a script the harness
    undertook to leave. A run whose geometry, mesh, patches and audits are all clean
    fails on it. T1 is exactly that run.

    So each completed case is checked twice and both verdicts are reported: **as
    shipped**, which is what the desk actually got, and **with I5's promise kept**, which
    is what the same mesh scores once `build.py` is where the brief says it is. Neither
    number is dropped, and the second is not used to launder the first.
    """
    from openreynolds import cad

    prompts = load_prompts()
    cfg, backend, client, instance_id, store = open_workspace(instance, local=local, work=work)
    try:
        sync_toolbox(backend)
        for name, prompt in prompts.items():
            if only and name not in only:
                continue
            rec = load_record("desk", name)
            if not rec or not rec.get("completed"):
                continue
            case_dir = rec["run"].get("case_dir", "")
            script = (artifact_dir("desk", name) / "build.py")
            if not script.exists():
                rec["recheck"] = {"ran": False, "why": "the run left no accepted script"}
                save_record("desk", name, rec)
                continue
            print(f"recheck/{name} ...", flush=True)
            backend.put_file(f"{case_dir}/build.py", script.read_bytes())
            verdict = cad.verify(backend, case_dir, f"accept-{name.lower()}",
                                 prompt["request"])
            rec["recheck"] = {"ran": True, "check": _check_digest(verdict)}
            save_record("desk", name, rec)
            print(f"  ok={verdict.ok} missing={verdict.missing}")
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


def report() -> int:
    """Everything on disk, as one table. No run, no model call, no instance."""
    prompts = load_prompts()
    rows = []
    for name in prompts:
        row = {"prompt": name}
        for arm in ("desk", "bash", "old", "replay", "oldcheck"):
            rec = load_record(arm, name)
            if not rec:
                continue
            run = rec.get("run", {})
            row[arm] = {
                "completed": rec.get("completed"),
                "ok": run.get("ok", run.get("check", {}).get("ok")),
                "seconds": rec.get("seconds", run.get("seconds")),
                "usd": run.get("usd"),
                "tokens": run.get("tokens"),
                "stopped": run.get("stopped"),
                "crashed": rec.get("crashed"),
            }
        rows.append(row)
    print(json.dumps(rows, indent=2, default=str))
    totals: dict[str, dict[str, float]] = {}
    for row in rows:
        for arm, cell in row.items():
            if arm == "prompt" or not isinstance(cell, dict):
                continue
            bucket = totals.setdefault(arm, {"seconds": 0.0, "usd": 0.0, "runs": 0})
            bucket["seconds"] += float(cell.get("seconds") or 0)
            bucket["usd"] += float(cell.get("usd") or 0)
            bucket["runs"] += 1
    print("\ntotals: " + json.dumps(totals, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=list(ARMS) + ["replay", "oldcheck", "recheck",
                                                      "bashcheck", "probes", "report"])
    parser.add_argument("--only", default="", help="comma-separated prompt names, e.g. T7,T8")
    parser.add_argument("--force", action="store_true", help="re-run one that already has a record")
    parser.add_argument("--instance", default=None)
    parser.add_argument("--local", action="store_true",
                        help="run against OpenFOAM on this machine instead of the image")
    parser.add_argument("--work", default=None, type=Path,
                        help="the local workspace directory (--local only)")
    args = parser.parse_args()

    only = [n.strip().upper() for n in args.only.split(",") if n.strip()]
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.what == "report":
        return report()
    if args.what == "probes":
        if args.local:
            # Saying no rather than doing something that looks like yes. C1's six
            # answers are already local measurements on OCP 7.9.3.1.1; the gate is
            # that they be re-measured on the image's 7.8.1, and running them here
            # again cannot satisfy it. See docs/cad-acceptance.md, "still owed".
            raise SystemExit(
                "probes --local would re-measure what C1 already measured on this "
                "machine's OCP; the gate is the image's. Run it without --local.")
        cfg, backend, client, instance_id, store = open_workspace(args.instance)
        try:
            out = run_probes(backend, RESULTS / "artifacts" / "probes")
        finally:
            client.close()
        print(out.get("compare", "")[-4000:])
        return 0
    if args.what == "replay":
        return drive_replay(only, args.force, args.instance, args.local, args.work)
    if args.what == "oldcheck":
        return drive_old_check(only, args.instance, args.local, args.work)
    if args.what == "recheck":
        return drive_recheck(only, args.instance, args.local, args.work)
    if args.what == "bashcheck":
        return drive_bash_check(only, args.instance, args.local, args.work)
    return drive(args.what, only, args.force, args.instance, args.local, args.work)


if __name__ == "__main__":
    raise SystemExit(main())
