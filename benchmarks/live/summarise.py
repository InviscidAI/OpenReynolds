#!/usr/bin/env python3
"""Turn the recorded results into `PROGRESS_LIVE.md`'s summary table.

    python benchmarks/live/summarise.py                 # print the table
    python benchmarks/live/summarise.py --write         # replace it in PROGRESS_LIVE.md

Reads every `benchmarks/live/results/*/result.json` that `modal_run.py` wrote and emits
one markdown row per graded check: the test, its target, the measured value, pass or
fail, the device, and the commit the code was at.

**Why this is a script and not a person with a keyboard.** The brief asks for a summary
table of every test with its target, its measured value and a commit hash, and the
obvious way to produce one is to read the terminal and type. This repository has a
directory -- `benchmarks/` -- whose entire premise is that a reference value transcribed
by hand drifts from the one the code applies, and `tests/test_benchmarks_index.py` exists
because that had already happened twice here. A summary table of 40-odd numbers typed
once is the same bet with worse odds. So the table is generated, the generator is
committed, and regenerating it is how it is checked.

The commit hash is read from `git rev-parse` at the time of **summarising**, which is not
quite the commit each run was made at. `result.json` carries no hash because the runner
cannot know one that is meaningful -- a run launched from a dirty tree has no commit. So
the row carries the hash the table was generated at and a dirty marker when the tree is
dirty, and that is said in the table's own header rather than implied.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO = HERE.parents[1]
PROGRESS = REPO / "PROGRESS_LIVE.md"
MARKER = "<!-- SUMMARY TABLE -->"

MILESTONE_OF = {
    "poiseuille": "M1", "cavity-re100": "M1", "cavity-re1000": "M1", "cylinder": "M1",
    "sphere": "M1", "crosscheck": "M1",
    "translating-sphere": "M2", "falling-sphere": "M2", "momentum": "M2",
    "falling-sphere-windowed": "M2",
    "ground-effect": "M3", "quadrotor": "M3", "underwater": "M3",
    "smoke": "M0", "benchmark-soa": "M4", "benchmark-aos": "M4",
}


def _commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True, timeout=60).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except Exception:
        return "unknown"


def _cell(text: object, limit: int = 150) -> str:
    """One markdown table cell: pipes escaped, newlines flattened, long prose cut.

    A target string in `cases.py` can run to a paragraph -- they carry the reference's
    interval and its provenance -- and a 400-character table cell is unreadable. Cut
    with an ellipsis rather than wrapped, because the full text is in the case's own
    report and in `result.json`, both of which the table points at.
    """
    flat = " ".join(str(text).split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _number(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return _cell(value, 60)


def _verdict(passed: object) -> str:
    """`PASS`, `FAIL` or `--`, from a verdict that may be a bool or a string of one.

    Tolerant on purpose. `Check.__post_init__` now coerces the verdict to a real `bool`
    before it is ever serialised, but results recorded before that fix carry the strings
    "True" and "False" -- and a summariser that cannot read the results already on disk
    is a summariser that makes you re-run an H100 to fix a typo.
    """
    if passed is None or passed == "None":
        return "--"
    if isinstance(passed, str):
        return "PASS" if passed.strip().lower() == "true" else "FAIL"
    return "PASS" if passed else "FAIL"


def rows() -> list[dict]:
    found = []
    if not RESULTS.is_dir():
        return found
    for path in sorted(RESULTS.glob("*/result.json")):
        label = path.parent.name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        device = str(data.get("gpu_name") or data.get("device") or "?")
        if data.get("error"):
            found.append({
                "label": label, "milestone": MILESTONE_OF.get(label, "--"),
                "check": "the run did not complete",
                "measured": "raised", "target": "a completed run", "verdict": "FAIL",
                "device": device, "seconds": data.get("seconds", 0.0),
            })
            continue
        for check in data.get("checks", []):
            verdict = _verdict(check.get("passed"))
            found.append({
                "label": label, "milestone": MILESTONE_OF.get(label, "--"),
                "check": check.get("name", ""),
                "measured": check.get("measured"),
                "target": check.get("target", ""),
                "verdict": verdict,
                "device": device, "seconds": data.get("seconds", 0.0),
            })
    return found


def table(found: list[dict], commit: str) -> str:
    if not found:
        return (f"{MARKER}\n\n_No results recorded yet. "
                f"`benchmarks/live/modal_run.py::m1` writes them into "
                f"`benchmarks/live/results/`, and "
                f"`python benchmarks/live/summarise.py --write` puts them here._\n")
    graded = [r for r in found if r["verdict"] in ("PASS", "FAIL")]
    failed = [r for r in graded if r["verdict"] == "FAIL"]
    lines = [
        MARKER,
        "",
        f"Generated by `python benchmarks/live/summarise.py --write` from "
        f"`benchmarks/live/results/*/result.json`, at commit **{commit}**. Every row's "
        f"device is the one that produced it. The commit is the one the table was "
        f"generated at, which is the nearest thing to a per-run hash that exists: a run "
        f"launched from a dirty tree has no commit, so the runner records none and this "
        f"marks the tree instead.",
        "",
        f"**{len(graded) - len(failed)} of {len(graded)} graded checks pass**, over "
        f"{len({r['label'] for r in found})} runs. Rows with no verdict are reported and "
        f"not graded, which the target column says in words.",
        "",
        "| run | M | check | target | measured | | device |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in found:
        lines.append(
            f"| `{row['label']}` | {row['milestone']} | {_cell(row['check'], 90)} "
            f"| {_cell(row['target'], 120)} | {_number(row['measured'])} "
            f"| **{row['verdict']}** | {_cell(row['device'], 40)} |")
    if failed:
        lines += ["", "Failing rows, in full:", ""]
        for row in failed:
            lines.append(f"- `{row['label']}`: {row['check']} measured "
                         f"{_number(row['measured'])} against {_cell(row['target'])}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help=f"replace everything after {MARKER} in PROGRESS_LIVE.md")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    found = rows()
    if args.as_json:
        print(json.dumps(found, indent=2, default=str))
        return 0
    rendered = table(found, _commit())
    if not args.write:
        print(rendered)
        return 0
    text = PROGRESS.read_text(encoding="utf-8")
    if MARKER not in text:
        print(f"{PROGRESS} has no {MARKER} to replace", file=sys.stderr)
        return 2
    PROGRESS.write_text(text[: text.index(MARKER)] + rendered, encoding="utf-8")
    print(f"wrote {len(found)} rows into {PROGRESS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
