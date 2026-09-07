"""The child interpreter: `cli.py build --script` run by path under `-I`, and its
files read back.

Stdlib only, and deliberately no kernel import: the parent process that hosts the desk
stays free of gmsh state, and the child is where every kernel module loads. That is
also why the result comes back as `result.json` (DESIGN.md 4.3), never as Python
objects. The child is launched by path with `-I` (D3): isolated mode drops the parent's
`sys.path` entries and `PYTHONPATH`, and the cli's `__main__` shim puts exactly the
checkout root back, so the kernel has one module name in the child on every platform.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RUN_TIMEOUT_S = 90
"""A build is 1-3 s; the limit is for a script that loops."""

ALLOWED_IMPORTS = {"math", "json"}
"""Plus the sketch API, which needs no import."""

CLI = Path(__file__).resolve().with_name("cli.py")
"""The child's entry point, run by path (D3)."""

PASS_THROUGH = ("HOME", "USERPROFILE", "TEMP", "TMP", "TMPDIR", "SYSTEMROOT", "APPDATA",
                "LOCALAPPDATA", "LANG", "LC_ALL")
"""Parent variables the child may see. Everything else -- and PYTHONPATH in particular --
stays behind; `-I` would ignore PYTHONPATH anyway, and the allowlist pins it twice."""

EXIT_CODES = {0, 2, 3, 4, 5}
"""0 built, lint clean; 2 lint errors; 3 the script raised; 4 timeout; 5 refused."""


@dataclass
class RunOutcome:
    rc: int
    """0 built, lint clean; 2 lint errors (picture drawn); 3 the script raised; 4 timeout;
    5 refused by the API (SketchError) or the guard (E-IMPORT, E-NO-SKETCH, two Sketches)."""
    text: str
    """What the model is shown: the print-back (0, 2, and 5 with a partial) or the refusal /
    trimmed traceback alone (3, and 5 without a partial) / the timeout line (4) --
    report.text's one rule."""
    png: bytes | None
    """preview.png when drawn (0, 2; 5 when the refusal carried a partial and the cli drew it)."""
    result: dict | None
    """result.json (section 4.3) when the child wrote one."""
    seconds: float

    @property
    def ready(self) -> bool:
        """rc == 0 and result["compliance"] is not None and no failing rows."""
        if self.rc != 0 or not self.result or self.result.get("compliance") is None:
            return False
        return not self.failing_claims

    @property
    def lint_clean(self) -> bool:
        if self.result is None:
            return self.rc == 0
        return not any(f.get("level") == "error" for f in self.result.get("lint") or [])

    @property
    def failing_claims(self) -> list[str]:
        table = (self.result or {}).get("compliance") or {}
        return [r["id"] for r in table.get("rows", []) if r.get("verdict") == "fail"]

    def as_dict(self) -> dict:
        return {"rc": self.rc, "text": self.text, "png": None if self.png is None else self.png.hex(),
                "result": self.result, "seconds": self.seconds}

    @staticmethod
    def from_dict(d: dict) -> "RunOutcome":
        png = d.get("png")
        return RunOutcome(rc=d["rc"], text=d.get("text", ""), png=None if png is None else bytes.fromhex(png),
                          result=d.get("result"), seconds=d.get("seconds", 0.0))


def command(script_path: Path, work: Path, claims_path: Path | None, reference: str | None,
            timeout_s: float = RUN_TIMEOUT_S) -> list[str]:
    """The child's argv: `python -I <cli.py> build --script ... --out ... --preview ...`
    plus `--claims` and `--reference` when given and `--timeout` so the child can set its
    own CPU limit where the platform has one."""
    argv = [sys.executable, "-I", str(CLI), "build", "--script", str(script_path), "--out", str(work),
            "--preview", str(work / "preview.png"), "--timeout", f"{timeout_s:g}"]
    if claims_path is not None:
        argv += ["--claims", str(claims_path)]
    if reference:
        argv += ["--reference", reference]
    return argv


def child_env(work: Path) -> dict[str, str]:
    """A pass-through allowlist, not three keys: PATH; PYTHONUTF8=1; MPLBACKEND=Agg;
    MPLCONFIGDIR=<work>/.mpl (created) so matplotlib never looks for a home directory;
    and, when present in the parent, HOME, USERPROFILE, TEMP, TMP, TMPDIR, SYSTEMROOT,
    APPDATA, LOCALAPPDATA, LANG, LC_ALL. Measured on this box: with {PATH, PYTHONUTF8,
    MPLBACKEND} alone `python -I -c "import gmsh, matplotlib.pyplot"` exits 1 with
    'RuntimeError: Could not determine home directory'; with MPLCONFIGDIR added it exits 0.
    PYTHONPATH is never passed (-I ignores it anyway; the test pins both)."""
    mpl = Path(work) / ".mpl"
    mpl.mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONUTF8": "1", "MPLBACKEND": "Agg",
           "MPLCONFIGDIR": str(mpl)}
    for key in PASS_THROUGH:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def timeout_text(timeout_s: float) -> str:
    return (f"the script ran past {timeout_s:g} s and was stopped; a loop in the script is not "
            "terminating")


def run_script(script: str, work: Path, claims_path: Path | None, reference: str | None,
               timeout_s: float = RUN_TIMEOUT_S) -> RunOutcome:
    """Writes work/script.py, runs [sys.executable, "-I", <geometry/cli.py>, "build",
    "--script", ..., "--out", work, "--preview", work/preview.png, "--claims", claims_path,
    "--reference", reference] with `child_env(work)`, capture_output, timeout. Reads
    work/result.json and work/preview.png. On TimeoutExpired: rc 4 and the text 'the script
    ran past 90 s and was stopped; a loop in the script is not terminating'."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for stale in ("result.json", "preview.png", "report.txt", "record.json"):
        # a lap that writes nothing must not be read as the previous lap's files
        try:
            (work / stale).unlink()
        except FileNotFoundError:
            pass
    script_path = work / "script.py"
    script_path.write_text(script, encoding="utf-8")
    argv = command(script_path, work, claims_path, reference, timeout_s)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout_s, env=child_env(work), cwd=str(work))
    except subprocess.TimeoutExpired:
        return RunOutcome(rc=4, text=timeout_text(timeout_s), png=None, result=None,
                          seconds=time.monotonic() - t0)
    seconds = time.monotonic() - t0

    result = None
    result_path = work / "result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result = None
    png_path = work / "preview.png"
    png = png_path.read_bytes() if png_path.exists() and png_path.stat().st_size > 0 else None

    rc = proc.returncode
    if result is not None and isinstance(result.get("report"), str):
        text = result["report"]
        rc = int(result.get("rc", rc))
    else:
        # every path the cli owns writes result.json, so a child that left none died in
        # the kernel (a crash, an unhandled exception): that is rc 3 whatever the exit
        # code was, never rc 2 "lint errors" with no findings to show
        tail = (proc.stdout + "\n" + proc.stderr).strip()[-3000:]
        text = (f"the child interpreter exited {rc} without a result.json; nothing in the script "
                f"to fix (reported to the desk)\n{tail}")
        rc = 3
    return RunOutcome(rc=rc, text=text, png=png, result=result, seconds=seconds)
