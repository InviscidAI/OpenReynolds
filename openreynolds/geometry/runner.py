"""The child interpreter: `cli.py build --script` run by path under `-I`, and its
files read back.

Stdlib only, and deliberately no kernel import: the parent process that hosts the desk
stays free of gmsh state, and the child is where every kernel module loads. That is
also why the result comes back as `result.json` (DESIGN.md 4.3), never as Python
objects. Skeleton (U0): `RunOutcome` and the constants; `run_script` and `child_env`
raise NotImplementedError until U4 lands.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_U4 = ("not built in the U0 skeleton: U4 (preview + runner + cli) implements it against "
       "DESIGN.md section 3.10")

RUN_TIMEOUT_S = 90
"""A build is 1-3 s; the limit is for a script that loops."""

ALLOWED_IMPORTS = {"math", "json"}
"""Plus the sketch API, which needs no import."""


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


def run_script(script: str, work: Path, claims_path: Path | None, reference: str | None,
               timeout_s: float = RUN_TIMEOUT_S) -> RunOutcome:
    """Writes work/script.py, runs [sys.executable, "-I", <geometry/cli.py>, "build",
    "--script", ..., "--out", work, "--preview", work/preview.png, "--claims", claims_path,
    "--reference", reference] with `child_env(work)`, capture_output, timeout. Reads
    work/result.json and work/preview.png. On TimeoutExpired: rc 4 and the text 'the script
    ran past 90 s and was stopped; a loop in the script is not terminating'."""
    raise NotImplementedError(_U4)


def child_env(work: Path) -> dict[str, str]:
    """A pass-through allowlist, not three keys: PATH; PYTHONUTF8=1; MPLBACKEND=Agg;
    MPLCONFIGDIR=<work>/.mpl (created) so matplotlib never looks for a home directory;
    and, when present in the parent, HOME, USERPROFILE, TEMP, TMP, TMPDIR, SYSTEMROOT,
    APPDATA, LOCALAPPDATA, LANG, LC_ALL. Measured on this box: with {PATH, PYTHONUTF8,
    MPLBACKEND} alone `python -I -c "import gmsh, matplotlib.pyplot"` exits 1 with
    'RuntimeError: Could not determine home directory'; with MPLCONFIGDIR added it exits 0.
    PYTHONPATH is never passed (-I ignores it anyway; the test pins both)."""
    raise NotImplementedError(_U4)
