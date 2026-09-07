"""The geometry command line: build | preview | check | reference | golden.

    python -I openreynolds/geometry/cli.py build   --script s.py [--claims c.json] --out DIR [--preview p.png] [--reference NAME]
                                                    (the child of runner.run_script; writes result.json, report.txt, record.json, preview.png)
    python -m openreynolds.geometry.cli preview    (--script | --record | --library NAME [--param k=v] | --spec spec.json [--scale]) --out p.png [--plain]
    python -m openreynolds.geometry.cli check      (--record geometry.json | --script s.py) [--claims c.json]   (lint + measure + compliance; exit 2 on errors)
    python -m openreynolds.geometry.cli reference  (prints api_summary(); `--write` rewrites reference.md)
    python -m openreynolds.geometry.cli golden     (--regenerate [NAME] | --check | --approve NAME --by "Kabir")

Exit codes: 0 built; 2 lint errors; 3 the script raised (and E-KERNEL-STATE); 4 timeout
(set by the parent); 5 refused (SketchError / guard / E-REFERENCE-UNAPPROVED). `build`
never writes a case: the case is written by `case.write_case` through `mesh2d.main`.

Run by path under `-I` (D3): the `__main__` shim puts the checkout root on `sys.path` and
imports `openreynolds.geometry.cli`, so the kernel has one module name in the child and
`openreynolds/trace.py` never shadows the stdlib `trace`. The shim sits above the
package-relative imports because, run by path, this file is `__main__` with no parent
package and a relative import would fail before any shim at the bottom ran. Skeleton
(U0): the entry points raise NotImplementedError until U4 lands; the shim is in place.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout root: `openreynolds` importable
    from openreynolds.geometry import cli as _cli                    # one module name for the kernel in the child
    sys.exit(_cli.main())

from . import claims, compile, library, lint, measure, preview, report, runner, sketch  # noqa: E402, F401

_U4 = ("not built in the U0 skeleton: U4 (preview + runner + cli) implements it against "
       "DESIGN.md section 3.15")


def exec_script(src: str, work: Path, claims_path: Path | None = None, reference: str | None = None,
                preview_path: Path | None = None) -> int:
    """`build --script`: the namespace is {"__name__": "__sketch__", **sketch.API} (the API
    is pre-bound: no import of the sketch module is ever needed, and both `from
    geometry.sketch import ...` and `from openreynolds.geometry.sketch import ...` are
    refused with the E-IMPORT text "the API needs no import"). The import guard is
    installed on `builtins.__import__` immediately before `exec(compile(src, "script.py",
    "exec"), ns)` and removed in a `finally`; it refuses only when the calling frame's
    globals carry `__name__ == "__sketch__"` and the name is not in
    `runner.ALLOWED_IMPORTS`. On Linux RLIMIT_AS 2 GB and RLIMIT_CPU = timeout + 5.
    Exactly one `Sketch` must exist (E-NO-SKETCH / E-MANY-SKETCHES); a traceback is
    trimmed to the frames whose filename is `script.py`, the failing line quoted, and a
    `TypeError` on a known constructor gets the signature hint (E-SCRIPT text). The rc-5
    path with a `partial` builds, measures and draws the single instance (3.15). Returns
    the exit code and writes result.json, report.txt, record.json, preview.png into `work`."""
    raise NotImplementedError(_U4)


def main(argv: list[str] | None = None) -> int:
    """The subcommands above; returns the exit code."""
    raise NotImplementedError(_U4)


if __name__ == "__main__":
    sys.exit(main())   # `python -m openreynolds.geometry.cli`: the package is known, the shim above did not run
