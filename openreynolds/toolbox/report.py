#!/usr/bin/env python3
"""A finished study as one PDF: the numbers, the figures, and the settings behind them.

`results.py` already writes `results.md` -- what was produced, what was skipped and
why, the headline numbers -- and that stays the working record. What it is not is a
document somebody outside the run can be handed: a markdown file plus a directory of
PNGs arrives as a directory of PNGs, and the settings that decide whether any of those
numbers mean anything are still spread over five dictionaries nobody opens. This
assembles the one file: cover, figures, methods, and what was asked for and not
produced. It complements `results.md` rather than replacing it, and it is written only
when it is asked for.

the writer is matplotlib

There is no PDF library on this instance and none is coming -- the network is sealed
(`ENVIRONMENT.md`). `matplotlib.backends.backend_pdf.PdfPages` is a multi-page PDF
writer that is already here, so every page is a matplotlib figure: text pages are text
drawn in a monospaced font at a fixed line pitch, and figure pages are the PNGs read
back through `imread`. That has one consequence worth knowing: a figure on a page of
this document is pixels, so the `reynolds-claims` chunk stamped in the original PNG
does NOT survive onto the page, the same way `gallery.py`'s contact sheet does not
carry it. The stamp is read from the original file on disk, before the page is drawn,
and what the page prints is the reading.

why the key numbers are not simply printed

An executive summary that prints a number beside a figure is the exact failure
`claims.py` was written for: in the backward-facing-step study the answer quoted 6.4,
the figure beside it was annotated 7.57, and nothing put the two side by side. So the
numbers on the cover of this document are not collected from whatever produced them
and set next to whatever picture looked relevant. Each one is tied to a figure or it is
not:

- **unverified** -- a number read out of the study's own files (a coefficient's tail
  mean, a final residual) with no stamped figure behind it. It is printed with the file
  it was read from, because a number nobody drew is still the answer somebody needs;
  what it is not is evidence.
- **verified** -- the value is stamped into a PNG that is still on disk, under the
  `reynolds-claims` keyword, by the same call that drew the annotation on it
  (`claims.caption`). The cover names the figure.
- **disagrees** -- a measured value and a stamped claim of the same quantity differ by
  more than `claims.DEFAULT_TOL`. Both numbers are printed, with the gap. This is F-36,
  and it is on the cover rather than in a log.

A number whose figure has been deleted since is unverified too, and says so.

In practice every number on a real study is the first of those three. Nothing in this
pipeline stamps a figure -- `claims.attach` is the only call that writes the chunk and
no renderer calls it, measured over 973 PNGs under `studies/` and `data/qa-runs/`, of
which zero carry one. So this page does not offer a verdict it cannot reach: where
nothing is stamped it says so once, in words rather than as a count of zero, drops the
per-row state column that would repeat one word down the page, and gives each number
the file it came from. The two stamped outcomes are kept because `claims.attach` is a
real call a study can make on purpose, and they come back the moment one does.

what it picks, and what it says about picking

The hero figure is chosen by a stated rule, printed on stdout and on the cover, and
`--hero` overrides it. Beyond that this picks nothing: the figures are grouped the way
`gallery.py` groups them, newest last, and a manifest row whose file is gone appears as
MISSING rather than quietly vanishing -- a study somebody tidied up should read as a
study somebody tidied up.

the template

`--template` takes a JSON file: the cover text, the fixed front matter, and the section
order. One is shipped beside this script as `report_template.json`. A `--template` that
is not there, or is not readable JSON, is refused by name rather than falling back to
the built-in default: a document that silently came out in the house default when a
company template was asked for is a document that gets sent.

what it is pointed at, and what gets printed of a field

A local study on a laptop is `studies/<id>/` with `files/` holding the mirrored
workspace tree, which itself nests as `files/<study-id>/<case>` (`store.py`). So
there are three places a person can reasonably point this at -- the study top, the
study home, the case -- and after #129 not one of them gave both the figures and the
numbers (#145): the top gave four blank pages and wrote a second
`.reynolds/manifest.jsonl` at the wrong level, the home gave the numbers and "No
artifact is registered in this study's manifest" because the manifest is one
directory down, and the case gave the numbers with all six figures MISSING while 30
PNGs sat one level up. `locate` is the answer: the study's shape is found rather than
guessed at, by looking down into `files/<id>/*/` for the `.reynolds/` as well as up,
and by resolving each manifest path against the directory those paths were written
relative to -- the state directory, or its parent, or its parent's parent. A
directory with no `.reynolds/` under it is not a study and does not get one written
into it.

The other half of the same incident (#139) is what a methods section prints of a
case: a `0/U` holding `internalField nonuniform List<vector> 139286(...)` was printed
verbatim, one 3.8 MB "line" folded at 100 characters, and the document came out at
1308 pages in 11 minutes with 1280 of those pages being three initial fields. 15
cases in the corpus have a `0/` file over 5,000 characters -- `pointLevel` from
snappyHexMesh, `alpha.water` from `setFields`, mapped `U`/`p`/`Phi`. So a field is
printed by its shape (`field_shape`), every value in the section is capped
(`MAX_VALUE_CHARS`), the section itself is capped (`METHODS_MAX_PAGES`), and in both
cases the page says it summarised rather than printed.

    python3 report.py /work/study
    python3 report.py /work/study --out ~/acme-report.pdf --json went-in.json
    python3 report.py /work/study --template report_template.json --logo acme.png
    python3 report.py /work/study --hero renders/vorticity.png --case cylinder
    python3 report.py /work/study --no-layers        # skip the wall-layer measurement
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402 - after the backend is fixed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claims  # noqa: E402  (sibling scripts, not a package)
import gallery  # noqa: E402
import log_digest  # noqa: E402
import mesh_digest  # noqa: E402
import preflight  # noqa: E402
import results  # noqa: E402
import study_state  # noqa: E402

PDF_NAME = "report.pdf"
TEMPLATE_NAME = "report_template.json"

SOURCE_TAG = "report-pack"
"""Written into the meta of the row this script registers, and skipped when the next
run reads the manifest back -- `gallery.py`'s arrangement, for its reason: without it
the second report would carry a page about the first one."""

REPORT_KIND = "report"
"""The manifest kind. `study_state.KINDS` already has it, and `results.py` registers
`results.md` under it: both are "the study written up", and a reader asking for the
report of a study wants whichever is newer, not a vocabulary lesson. No new kind."""

PAGE_INCHES = (8.27, 11.69)  # A4 portrait, because this gets printed
PAGE_DPI = 110

BODY_LINES = 56
"""Monospaced lines that fit on a text page at `BODY_SIZE`. Measured by filling one."""
BODY_SIZE = 7.6
LINE_PITCH = 0.0152

FIGURES_PER_PAGE = 4

MAX_VALUE_CHARS = 200
"""The longest a single value read out of a case may be on a page of this document.

#139: a `0/U` written by a mapped start carries its 139286 values as one
`internalField` entry, 3.8 MB of it, and nothing between the file and the page had a
width. Two page-lines is the cap, and a capped line says it was capped -- a reader
who cannot see that something was left out cannot go and read the file."""

METHODS_MAX_PAGES = 12
"""And the section as a whole, for the same incident: three capped values still make
a long document on a case with 40 fields in `0/`, and 1280 pages of methods took 11
minutes to draw. A methods section longer than this is not a methods section."""

NEGLIGIBLE_FRACTION = 1e-9
"""Below this much of the record's own largest value, a coefficient is zero.

`Cs (last): 3.2709e-20` went on a cover page as a key number. On a symmetric body the
side force is zero and the solver's answer is round-off; printing the round-off as a
headline reads as a measurement of something."""

HERO_KINDS = (
    "velocity", "pressure", "vorticity", "streamlines",
    "forces", "residuals", "mesh-full", "geometry-preview",
)
"""The fallback order for a hero when no figure carries a claim: the picture of the
answer before the picture of the method. It is a preference, not a judgement about any
particular study, which is why the rule that was used is printed."""

DEFAULT_TEMPLATE: dict[str, Any] = {
    "title": "",
    "subtitle": "",
    "front_matter": [],
    "sections": ["summary", "figures", "methods", "gaps"],
    "footer": "Numbers as produced. Whether they are converged, mesh independent or "
              "right is not decided here.",
}

SECTIONS = ("summary", "figures", "methods", "gaps")


# -- the template ------------------------------------------------------------------


class Refused(Exception):
    """An argument that would have produced a plausible-looking wrong document."""


def load_template(path: Path | str | None) -> tuple[dict[str, Any], list[str]]:
    """The template, and any notes about it worth printing.

    Returns the built-in default when nothing was asked for, and raises when something
    was asked for and is not there. The asymmetry is the whole point: the default is
    what you get when you did not ask, never what you get instead of what you asked
    for.
    """
    if path is None:
        return dict(DEFAULT_TEMPLATE), []
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refused(
            f"refused: no template at {path}.\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            "A --template names the cover text, the fixed front matter and the section\n"
            "order. Falling back to the built-in default here would write a document in\n"
            "the house style under the name of the one you asked for, and nothing on the\n"
            "page would say so. There is an example beside this script at "
            f"{TEMPLATE_NAME}."
        ) from exc
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Refused(
            f"refused: the template at {path} is not readable JSON.\n\n"
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(loaded, dict):
        raise Refused(
            f"refused: the template at {path} is a {type(loaded).__name__}, not an "
            "object.\n\nIt wants keys: title, subtitle, front_matter, sections, footer."
        )
    template = dict(DEFAULT_TEMPLATE)
    template.update(loaded)
    notes: list[str] = []
    wanted = [str(name) for name in template.get("sections") or []]
    unknown = [name for name in wanted if name not in SECTIONS]
    if unknown:
        notes.append(
            f"template section(s) {', '.join(unknown)} are not ones this writes; "
            f"known: {', '.join(SECTIONS)}"
        )
    kept = [name for name in wanted if name in SECTIONS]
    if not kept:
        kept = list(DEFAULT_TEMPLATE["sections"])
        notes.append("the template named no section this writes, so all four are written")
    template["sections"] = kept
    return template, notes


# -- finding the study a person pointed at -------------------------------------------


_PROCESSOR = re.compile(r"^processors?\d+(_\d+-\d+)?$")
"""A decomposed run's per-rank directories. Each has a `constant/polyMesh`, so
`study_state.is_case_dir` says yes to all four of them and `discover_cases` returned
"the first of 6 cases" for a study with one case (#145). They are one case."""


def locate(given: Path | str) -> dict[str, Any]:
    """Where this study's state is, and what its manifest paths are relative to.

    #145: a mirrored study is `studies/<id>/files/<id>/<case>/.reynolds/` with the
    manifest's paths written relative to `files/<id>/`, and `study_state.find_root`
    only ever walks up. Pointed at the study top it found no state at all and wrote a
    second one; pointed at the case it found the state and resolved every figure one
    directory too low. So the state directory is looked for downwards as well, and the
    base that manifest paths resolve against is chosen by how many of them it actually
    finds on disk rather than assumed to be the state directory.

    Returns `state_root` (the directory whose `.reynolds/` holds the state), `base`
    (what manifest paths are relative to), `bases` (every base tried, closest first),
    and `is_study` -- false when nothing under the given path has a `.reynolds/` at
    all, which is the one case where nothing may be written.
    """
    given = Path(given)
    if not given.exists():
        # #186: every path below treats a missing directory as an empty one, so a
        # mistyped study wrote a 4-page document saying "no case directory under the
        # study home" and exited 0 -- or, pointed at a name with nothing above it,
        # came out of `choose_case`'s `study.iterdir()` as a traceback and exit 1.
        # Neither is an answer about a run. A path that is not there is refused by
        # name, the way a --template that is not there already is.
        raise Refused(
            f"refused: nothing at {given}.\n\n"
            "A document is written about a study on this disk. There is no directory "
            "of that name here, so there is nothing to read and nothing to say about "
            "it: a document written anyway would carry the house default template "
            "under the name of a study nobody can open."
        )
    notes: list[str] = []
    state = study_state.STATE_DIR
    up = study_state.find_root(given)
    is_study = (up / state).is_dir()
    state_root = up
    if not is_study:
        # Downwards, in the order a study nests: the study home's case, then the
        # mirrored `files/<id>` and its case.
        for pattern in (f"*/{state}", f"files/*/{state}", f"files/*/*/{state}"):
            found = sorted(path for path in given.glob(pattern) if path.is_dir())
            if found:
                state_root, is_study = found[0].parent, True
                notes.append(f"the study's state is under {state_root}, not at {given}")
                break

    if not is_study and not study_state.discover_cases(state_root):
        # No `.reynolds/` anywhere under here, which is 11 of 16 real studies, and no
        # case at this level either: `discover_cases` looks one level down and a
        # mirrored study keeps its cases two, at `files/<id>/<case>`. So `choose_case`,
        # `measured_numbers` and `gaps` were all handed a directory with nothing under
        # it and the documented entry point wrote a complete-looking document with no
        # numbers and no methods in it -- 4 pages against the 6 the same tool gives
        # when pointed one level down, and 13 numbers against none (#179).
        #
        # `compare.py:665` descends the same way for the same layout. Only when there
        # is no case here at all, so a study whose cases sit at the top is untouched,
        # and `is_study` stays false either way -- there is still no state directory
        # here and nothing may be written.
        for pattern in ("files/*", "files/*/*"):
            found = sorted(path for path in state_root.glob(pattern)
                           if path.is_dir() and study_state.discover_cases(path))
            if found:
                notes.append(f"no .reynolds/ and no case under {state_root}, so the "
                             f"case(s) were read from {found[0]} -- the mirrored layout "
                             f"keeps them at files/<id>/<case>")
                state_root = found[0]
                break

    bases = [state_root]
    for parent in (state_root.parent, state_root.parent.parent):
        # The same guard `find_root` uses walking up: `/work` is every study's
        # neighbour, not any study's home, and the filesystem root is nobody's.
        if parent in bases or parent == parent.parent or parent.name == "work":
            break
        bases.append(parent)

    rows = study_state.artifacts(root=state_root, exists=False) if is_study else []
    base = state_root
    if rows:
        scored = [(sum(1 for row in rows if (candidate / str(row["path"])).exists()), -index,
                   candidate) for index, candidate in enumerate(bases)]
        best = max(scored)
        base = best[2]
        if base != state_root:
            notes.append(f"{best[0]} of {len(rows)} manifest path(s) resolve under "
                         f"{base}, not under the state directory {state_root}")
    return {"given": str(given), "state_root": state_root, "base": base,
            "bases": bases, "is_study": is_study, "notes": notes}


def resolve_path(relative: str, bases: list[Path]) -> Path:
    """A manifest path against the first base that has it, else the nearest base.

    Per row rather than per manifest: a study restarted in a different layout has rows
    of both kinds in one file, and a row that resolves nowhere still has to point
    somewhere for the MISSING panel to name."""
    for base in bases:
        candidate = base / relative
        if candidate.exists():
            return candidate
    return bases[0] / relative


def manifest_rows(
    state_root: Path, bases: list[Path], case: str = ""
) -> list[dict[str, Any]]:
    """The manifest, with every `abspath` resolved through `resolve_path`.

    `study_state.artifacts` resolves against the state directory, which is the
    assumption #145 is about; this is the same list with that one thing corrected.
    """
    rows = []
    for row in study_state.artifacts(root=state_root, case=case, exists=False):
        row = dict(row)
        row["abspath"] = str(resolve_path(str(row.get("path", "")), bases))
        rows.append(row)
    return rows


# -- the case the methods section is about -------------------------------------------


def _offered(cases: list[Path]) -> str:
    """The case names a refused `--case` could have had, for the refusal to name."""
    if not cases:
        return ("No case directory was found under the study either, so there is no "
                "name to offer: `--case` with none of them there would describe "
                "nothing.")
    return ("The case(s) under this study: "
            + ", ".join(sorted(case.name for case in cases))
            + ". Leaving --case off takes the first of them.")


def choose_case(study: Path, wanted: str = "") -> tuple[Path | None, str, list[str]]:
    """The case directory the methods are read from, the rule, and the ones not used.

    A study can hold several cases, and a methods section built from two of them at
    once would describe neither. So one is chosen and the rest are named on the page:
    "this is the mesh and the settings of `cylinder`" is a document, "these are some
    settings from somewhere in the study" is not.
    """
    # `study_state.discover_cases` drops `processor*` itself, for every caller. The
    # ranks are counted here anyway, because the note below is worth saying and the
    # count is no longer recoverable from what it returns.
    cases = [case for case in study_state.discover_cases(study)
             if not _PROCESSOR.match(case.name)]
    seen = {case.name for case in cases}
    dropped = sum(1 for child in study.iterdir()
                  if child.is_dir() and _PROCESSOR.match(child.name)
                  and child.name not in seen)
    # Said on the page rather than silently dropped: four processor directories are
    # evidence the run was decomposed, which is worth a reader's attention, and
    # `--case processor2` describing one rank's slice as the study's methods is not.
    ranks = (f"; its {dropped} processor* "
             + ("directory is" if dropped == 1 else "directories are")
             + " the same case, not others") if dropped else ""
    if wanted:
        if _PROCESSOR.match(wanted):
            # #145 noted this would be honoured: a rank's slice described as the
            # study's methods is a methods section about a quarter of a mesh.
            raise Refused(
                f"refused: --case {wanted} names one rank of a decomposed case, not a "
                "case; the case is the directory the processor* ones sit in.\n\n"
                + _offered(cases)
            )
        for case in cases:
            if case.name == wanted:
                return case, f"--case {wanted}", [c.name for c in cases if c != case]
        if (study / wanted).is_dir():
            return (study / wanted), f"--case {wanted}", [c.name for c in cases]
        # #186: this used to be a rule printed on the page, so a mistyped --case
        # wrote a whole document with no methods and no numbers in it and exited 0 --
        # while `--template nothere.json` was refused with exit 2 on the same study.
        # A case name that was typed and is wrong is the same mistake as a template
        # name that was typed and is wrong, and the asymmetry decided nothing: a
        # document that silently came out in the house default is a document that
        # gets sent.
        raise Refused(
            f"refused: --case {wanted} names no directory under {study}.\n\n"
            + _offered(cases)
        )
    if not cases:
        return None, "no case directory under the study home", []
    if len(cases) == 1:
        return cases[0], f"the study's only case ({cases[0].name}){ranks}", []
    return cases[0], (f"the first of {len(cases)} cases ({cases[0].name}); "
                      f"name another with --case{ranks}"), [c.name for c in cases[1:]]


# -- the numbers, each one tied to a figure or not ------------------------------------


def stamped_claims(
    state_root: Path, bases: list[Path], base: Path, case: str = ""
) -> list[dict[str, Any]]:
    """Every claim this study has, with whether its figure is still readable.

    The two sources are `claims.collect`'s, for its reasons -- the manifest misses a
    figure drawn before the study had state, the images miss nothing but have to be
    found -- gathered here rather than there only because of #145: a manifest row's
    figure resolves through `resolve_path`, and the PNGs are walked from the base the
    study's paths are relative to, which is where the 30 pictures of
    `20260829-100024-2825` actually are.

    What this adds is the one thing a cover page has to know and neither source states
    -- whether the number can still be read out of the picture's own bytes right now.
    A claim whose figure has been deleted, or redrawn by something that did not stamp
    it, is a claim with nothing behind it any more.
    """
    found: dict[tuple[str, str], dict[str, Any]] = {}

    def take(figure: Path, rows: Iterable[dict[str, Any]]) -> None:
        for one in rows:
            if not isinstance(one, dict) or "value" not in one:
                continue
            entry = dict(one)
            entry["figure"] = str(figure)
            found[(str(figure), str(entry.get("name", "")))] = entry

    for row in study_state.artifacts(root=state_root, kind=claims.CLAIM_KIND,
                                     case=case, exists=False):
        meta = row.get("meta") or {}
        take(resolve_path(str(row.get("path", "")), bases), meta.get("claims") or [])
    for png in sorted(base.rglob("*.png")):
        if study_state.STATE_DIR in png.parts:
            continue
        take(png, claims.read_png(png))

    out: list[dict[str, Any]] = []
    for key in sorted(found):
        figure = Path(str(found[key].get("figure", "")))
        entry = dict(found[key])
        entry["figure_exists"] = figure.is_file()
        entry["in_png"] = bool(claims.read_png(figure)) if entry["figure_exists"] else False
        out.append(entry)
    return out


def is_steady_solver(executable: str) -> bool:
    """Whether a solver is a steady one, from its name.

    Copied from `openreynolds/convergence.py` (`is_steady_solver`), which is above the
    toolbox and not importable when these scripts are run out of `/work/.toolbox/`.
    The reason it is needed here: `force record: 500 rows, t = 1 to 500` went on a
    cover page for a `simpleFoam` run whose `Time` column is an iteration count, not
    seconds. `spectra.py` was being fixed for the same confusion in the same sweep.
    """
    return "simple" in (executable or "").lower()


def zero_the_negligible(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coefficients that are zero to within the record's own scale, reported as zero.

    `Cs (last): 3.2709e-20` was printed as a key number. The side-force coefficient of
    a symmetric body is zero and what the solver wrote is its round-off; a cover page
    that quotes the round-off to five figures has made a measurement out of nothing.
    The scale is the record's own largest value, so nothing here depends on a unit or
    on what sort of coefficient it is, and the original value is still printed -- this
    says the number is zero, it does not hide it.

    Only the force record's own rows are passed in. A final residual of 1e-8 is small
    against a Cd of 1.2 and is not negligible: it is the answer.
    """
    values = {id(row): _number_in(row["text"]) for row in rows}
    scale = max((abs(value) for value in values.values() if value is not None), default=0.0)
    if scale <= 0.0:
        return rows
    for row in rows:
        value = values.get(id(row))
        if value is None or value == 0.0 or abs(value) >= NEGLIGIBLE_FRACTION * scale:
            continue
        row["negligible"] = True
        row["text"] = (f"0 -- negligible, not a headline (the file says {value:.5g}, "
                       f"under {NEGLIGIBLE_FRACTION:g} of this record's largest value "
                       f"{scale:.5g})")
    return rows


def measured_numbers(case: Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    """The numbers the case's own files carry: the coefficients and the reference
    values behind them, and where the residuals ended up.

    These come from the files rather than from a claim, so none of them arrives with a
    figure attached. `tie_numbers` is what decides whether any of them has one.
    """
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    if case is None:
        return rows, ["no case directory, so no coefficients or residuals were read"]

    # A steady solver's `Time` column counts iterations, not seconds, so the force
    # record's "t = 1 to 500" named the wrong axis on every steady run.
    application = preflight.parse_control(
        read_dict(case, "system/controlDict")).get("application", "")
    steady = is_steady_solver(application)
    try:
        grouped = results.find_force_files(case)
        if grouped:
            key = results.choose_force_series(grouped)
            history = results.read_history(grouped[key])
            columns = results.plot_columns(history["columns"])
            force_rows: list[dict[str, Any]] = []
            for label, value in results.force_headline(history, columns).items():
                text = str(value)
                if steady:
                    text = text.replace("t = ", "iterations ")
                force_rows.append({"label": label, "text": text, "source": f"{key}"})
            rows.extend(zero_the_negligible(force_rows))
            if steady:
                notes.append(f"{key} comes from a steady run ({application}), so its "
                             "Time column is counted in iterations, not seconds")
            for line in history.get("superseded", []) + history.get("ignored", []):
                notes.append(f"force record: {line}")
        else:
            notes.append("no force or coefficient .dat under postProcessing/")
    except Exception as exc:  # noqa: BLE001 - one missing number, not a missing document
        notes.append(f"coefficients not read: {type(exc).__name__}: {exc}")

    log = results.find_solver_log(case)
    if log is None:
        notes.append("no solver log in the case, so no final residuals")
    else:
        try:
            data = log_digest.digest(log)
            shape = log_digest.residual_shape(data.get("residuals") or {})
            for field, last in sorted(shape.get("last", {}).items()):
                rows.append({
                    "label": f"final residual {field}",
                    "text": f"{last:.3g} ({shape['fields'].get(field, '?')})",
                    "source": log.name,
                })
        except Exception as exc:  # noqa: BLE001
            notes.append(f"residuals not read: {type(exc).__name__}: {exc}")
    return rows, notes


def _number_in(text: str) -> float | None:
    """The one number in a formatted value, when there is exactly one to read.

    `claims._NUMBER` rather than a fresh regex, deliberately: it is the pattern that
    refuses a percentage, and the reasoning for that -- "within 5% of Gartling's 6.1"
    offering 5.0 as the value -- is written out beside it. A second pattern here would
    be a second chance to get that wrong.
    """
    found = claims._NUMBER.findall(str(text))
    if len(found) != 1:
        return None
    try:
        return float(found[0])
    except ValueError:
        return None


def tie_numbers(
    figure_claims: list[dict[str, Any]],
    measured: list[dict[str, Any]],
    tol: float = claims.DEFAULT_TOL,
) -> list[dict[str, Any]]:
    """Every number for the cover, each with the figure behind it or the reason there
    is none.

    This is the function the whole script exists for. `claims.py` says why: a claim and
    the artefact that is supposed to support it are produced by different code, and
    nothing ever puts the two numbers side by side. Here they are side by side before
    either reaches a page, and the three outcomes are kept apart on purpose --
    `verified` carries a figure, `disagrees` carries two numbers and the gap between
    them, `unverified` carries a number and the admission that nothing drew it.

    Matching is by spelling, through `claims.searchable`: a measured label contains one
    of the claim's own spellings, folded for case. Deliberately weak in the same
    direction `claims.check` is weak -- a claim nobody can match reads as unverified,
    which is true, rather than as agreement.
    """
    out: list[dict[str, Any]] = []
    for row in figure_claims:
        figure = Path(str(row.get("figure", "")))
        entry = {
            "label": str(row.get("name") or row.get("symbol") or "claim"),
            "text": claims.caption(row),
            "value": float(row["value"]),
            "units": str(row.get("units") or ""),
            "figure": str(figure),
            "source": str(row.get("source") or ""),
        }
        if row.get("in_png"):
            entry["state"] = "verified"
            entry["why"] = f"stamped in {figure.name}"
        elif row.get("figure_exists"):
            entry["state"] = "unverified"
            entry["why"] = (f"{figure.name} is on disk and carries no claim stamp "
                            "(redrawn by something that did not stamp it)")
        else:
            entry["state"] = "unverified"
            entry["why"] = f"the figure it was drawn from is gone ({figure})"
        out.append(entry)

    for row in measured:
        label = str(row["label"])
        folded = label.casefold()
        value = _number_in(row["text"])
        entry = {
            "label": label,
            "text": str(row["text"]),
            "value": value,
            "units": "",
            "figure": "",
            "source": str(row.get("source") or ""),
            "state": "unverified",
            # What this row knows is where the number was read, so that is what it
            # says. It used to say "no figure is stamped with this number", which is
            # true of every row on every real study -- nothing in this pipeline
            # stamps -- so printing it per row spent twenty-two lines repeating one
            # fact that `stamp_backing` states once, and spent them in place of the
            # provenance each number actually has.
            "why": (f"read from {row['source']}" if row.get("source")
                    else "read from the study's own files"),
            # Carried, not recomputed: `zero_the_negligible` decided this against the
            # force record's own scale, and the cover orders the page by it.
            "negligible": bool(row.get("negligible")),
        }
        for claimed in figure_claims:
            if not claimed.get("in_png"):
                continue
            if not any(word.casefold() in folded for word in claims.searchable(claimed)):
                continue
            figure = Path(str(claimed.get("figure", "")))
            if value is None:
                entry["why"] = (f"{figure.name} claims this quantity; this value is not "
                                "a single number, so nothing was compared")
                break
            gap = claims.apart(value, float(claimed["value"]))
            entry["figure"] = str(figure)
            if gap <= tol:
                entry["state"] = "verified"
                entry["why"] = f"agrees with {figure.name} ({claims.caption(claimed)})"
            else:
                entry["state"] = "disagrees"
                entry["why"] = (f"{figure.name} was drawn from "
                                f"{float(claimed['value']):.6g}, this reads {value:.6g} "
                                f"-- {gap * 100:.0f}% apart")
            break
        out.append(entry)
    return out


# -- the hero ------------------------------------------------------------------------


def pick_hero(
    entries: list[gallery.Entry],
    figure_claims: list[dict[str, Any]],
    given: Path | None = None,
) -> tuple[Path | None, str]:
    """The one figure on the cover, and the rule that chose it.

    Returned together because the rule is part of the answer. A document that puts one
    picture at the front has made an editorial decision, and a reader who cannot see
    which decision cannot disagree with it. `--hero` is the override, and it is honoured
    whether or not the file is registered in the manifest.
    """
    if given is not None:
        return Path(given), "given on the command line with --hero"

    stamped = [row for row in figure_claims if row.get("in_png")]
    if stamped:
        best = max(stamped, key=lambda row: len(claims.read_png(Path(str(row["figure"])))))
        return (Path(str(best["figure"])),
                "the figure carrying the most stamped claims -- a picture with the "
                "answer written into it")

    present = [entry for entry in entries if gallery.image_for(entry) is not None]
    for kind in HERO_KINDS:
        of_kind = [entry for entry in present if entry.kind == kind]
        if of_kind:
            return (gallery.image_for(of_kind[-1]),
                    f"no figure carries a claim, so the newest '{kind}' artifact, "
                    f"first of {', '.join(HERO_KINDS)} to appear in this study")
    if present:
        return (gallery.image_for(present[-1]),
                f"no figure carries a claim and none is of a preferred kind, so the "
                f"newest registered picture ({present[-1].kind})")
    return None, "the study has registered no picture that is still on disk"


# -- the methods material -------------------------------------------------------------


def read_dict(case: Path | None, relative: str) -> str:
    if case is None:
        return ""
    try:
        return (case / relative).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def as_written(body: str) -> list[str]:
    """A dictionary block's own lines, comments gone, blanks gone, nothing parsed.

    Used where the keys are not words. `preflight.entry_values` reads `key value;`
    pairs, and a `divSchemes` block's keys are `div(phi,U)` -- so the pattern skips
    past the key it cannot match and reports the first bare word it finds inside the
    value, which came out as `bounded  Gauss limitedLinear 1`. The divergence scheme is
    one of the few entries in a case that changes the answer on its own, so it is
    quoted rather than parsed: the block as the case writes it is exactly right here,
    and is what somebody reproducing the run would copy.
    """
    return [line.strip() for line in preflight.strip_comments(body).splitlines()
            if line.strip()]


def turbulence(case: Path | None) -> tuple[str, str, str]:
    """`simulationType`, the model, and the file it was read from.

    Two files, because OpenFOAM moved the entry: `constant/momentumTransport` in the
    .com line, `constant/turbulenceProperties` in the ESI one that is on this image.
    `ladder.py` reads both for the same reason.
    """
    for name in ("constant/turbulenceProperties", "constant/momentumTransport"):
        text = read_dict(case, name)
        if not text:
            continue
        values = preflight.entry_values(text)
        kind = values.get("simulationType", "")
        model = ""
        for keyword, entry in (("RAS", "RASModel"), ("LES", "LESModel")):
            body = preflight.block_body(text, keyword)
            if body:
                inner = preflight.entry_values(body)
                model = inner.get(entry) or inner.get("model") or ""
                if model:
                    break
        if kind or model:
            return kind, model, name
    return "", "", ""


_WORD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:\-]*$")
"""An OpenFOAM word, which is what a patch name and a boundary-condition type both
are. Used to tell one from a chunk of a binary payload (#185)."""


def boundary_types(case: Path | None, field: str) -> tuple[list[tuple[str, str]], str]:
    """`(patch, type)` out of one `0/` field's `boundaryField`, and a note if any of
    it would not read.

    The boundary conditions are the case, far more than the solver name is: the same
    mesh with `fixedValue` and with `zeroGradient` on the outlet are two different
    problems with two different answers, and nothing else in the document would say
    which was solved.

    This is #185. `field_shape` beside it was hardened for #139 and this was not: a
    `format binary` field writes its nonuniform patch values as raw IEEE-754 bytes
    inside the `boundaryField` block, and those bytes contain braces and semicolons,
    so the text parser walked out of one patch entry and reported two more. On
    `20260830-050338-cfd5`'s `0/thickness` -- 6 patches in
    `constant/polyMesh/boundary` -- it returned 8, the last two named
    `Yu?J\\xef\\xbf\\xbdNQu?` and the like, and both reached page 5 of the PDF with 11
    control characters in them (matplotlib: "Glyph 6 missing from font"). `_capped`
    caps how much of such a name is printed, not whether it is a name. A patch name
    and a type are both OpenFOAM words; a slice of a double is not, so what cannot be
    a patch is counted and named as unread rather than printed as one.
    """
    text = read_dict(case, f"0/{field}")
    if not text:
        return [], ""
    body = preflight.block_body(text, "boundaryField")
    rows = []
    unread = 0
    for name, inner in preflight.block_entries(body):
        if not name:
            continue
        kind = preflight.entry_values(inner).get("type", "?")
        if not _WORD.match(name) or not (kind == "?" or _WORD.match(kind)):
            unread += 1
            continue
        rows.append((name, kind))
    if not unread:
        return rows, ""
    fmt = preflight.entry_values(preflight.block_body(text, "FoamFile")).get("format", "")
    why = (f"the file is `format {fmt}`, whose patch values are raw bytes to a text "
           "parser" if fmt and fmt != "ascii"
           else "its text did not parse as a dictionary")
    return rows, (f"{unread} more entr{'y' if unread == 1 else 'ies'} in this "
                  f"boundaryField did not read as a patch ({why}); "
                  "constant/polyMesh/boundary is the list of patches this mesh has")


_NONUNIFORM = re.compile(r"List<(\w+)>\s*(\d+)?")


def field_shape(text: str) -> str:
    """One `0/` field as its shape: class, dimensions, and what its `internalField`
    is -- never the payload.

    This is #139. The old line was `internalField {value}` with the value as written,
    and a mapped, restarted, `setFields`-ed or snappyHexMesh-ed `0/` writes that value
    as `nonuniform List<vector> 139286((1.03809 -0.974742 0) ...)`: 3.8 MB on one
    line, folded to 100 characters a page-line, 1280 pages of methods, 11 minutes.
    What a methods section wants from a field is the initial condition, and the
    initial condition of a nonuniform field is that it is nonuniform and how many
    values it has. The contents are the case, which the case already is.
    """
    header = preflight.entry_values(preflight.block_body(text, "FoamFile"))
    values = preflight.entry_values(preflight.outer_text(text))
    internal = values.get("internalField", "")
    if "nonuniform" in internal:
        match = _NONUNIFORM.search(internal)
        kind = match.group(1) if match else "?"
        count = (f"{int(match.group(2)):,} entries"
                 if match and match.group(2) else "entry count not stated")
        shape = (f"nonuniform List<{kind}>, {count} -- summarised, not printed "
                 f"({len(internal):,} characters)")
    elif len(internal) > MAX_VALUE_CHARS:
        shape = (f"{internal[:MAX_VALUE_CHARS]} ... -- summarised, not printed "
                 f"({len(internal):,} characters)")
    else:
        shape = internal or "not stated"
    bits = [header.get("class", "") or "class not stated"]
    if values.get("dimensions"):
        bits.append(values["dimensions"])
    return ", ".join(bits) + f", internalField {shape}"


def find_check_log(case: Path | None) -> Path | None:
    """The newest `checkMesh` output in the case, whatever it was called.

    `log.checkMesh` is the convention and `checkMesh.log` is what half the scripts in
    the world write, so both are looked for rather than one being declared correct.
    """
    if case is None:
        return None
    found = [p for p in case.glob("*") if p.is_file() and "checkmesh" in p.name.casefold()]
    if not found:
        found = [p for p in case.glob("log*") if p.is_file()
                 and "Checking geometry" in p.read_text(errors="replace")[:20000]]
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def methods_lines(
    case: Path | None,
    case_rule: str,
    others: list[str],
    *,
    layers: bool = True,
) -> tuple[list[str], list[str]]:
    """The methods section as lines, and the files it was read from.

    Only the settings that change an answer. A methods section that reproduced every
    entry of every dictionary would be a copy of the case, which the case already is;
    what a reader needs is the solver, the schemes, the turbulence model, the time
    controls, the convergence criteria, the boundary conditions, the mesh it ran on,
    and the reference values any coefficient was divided by.
    """
    lines: list[str] = []
    read: list[str] = []
    lines.append(f"case: {case if case else 'none'}")
    lines.append(f"chosen by: {case_rule}")
    if others:
        lines.append(f"other cases in this study, not described here: {', '.join(others)}")
    lines.append("")

    control = preflight.parse_control(read_dict(case, "system/controlDict"))
    if control:
        read.append("system/controlDict")
        lines.append("solver and time controls (system/controlDict, top level only)")
        for key in ("application", "startFrom", "startTime", "stopAt", "endTime",
                    "deltaT", "writeControl", "writeInterval", "purgeWrite",
                    "adjustTimeStep", "maxCo", "maxDeltaT", "writeFormat"):
            if key in control:
                lines.append(f"  {key:<16} {control[key]}")
    else:
        lines.append("solver and time controls: system/controlDict was not readable")
    lines.append("")

    kind, model, where = turbulence(case)
    if where:
        read.append(where)
        lines.append(f"turbulence ({where})")
        lines.append(f"  simulationType   {kind or 'not stated'}")
        lines.append(f"  model            {model or 'none named'}")
    else:
        lines.append("turbulence: no turbulenceProperties or momentumTransport in the case")
    transport = preflight.entry_values(read_dict(case, "constant/transportProperties"))
    if transport:
        read.append("constant/transportProperties")
        lines.append("")
        lines.append("transport (constant/transportProperties, as written)")
        for key, value in sorted(transport.items()):
            lines.append(f"  {key:<16} {value}")
    lines.append("")

    schemes_text = read_dict(case, "system/fvSchemes")
    if schemes_text:
        read.append("system/fvSchemes")
        lines.append("discretisation (system/fvSchemes)")
        for block in ("ddtSchemes", "gradSchemes", "divSchemes", "laplacianSchemes",
                      "interpolationSchemes", "snGradSchemes"):
            body = preflight.block_body(schemes_text, block)
            if not body.strip():
                continue
            lines.append(f"  {block}")
            lines.extend(f"    {line}" for line in as_written(body))
    else:
        lines.append("discretisation: system/fvSchemes was not readable")
    lines.append("")

    solution_text = read_dict(case, "system/fvSolution")
    if solution_text:
        read.append("system/fvSolution")
        lines.append("linear solvers and convergence (system/fvSolution)")
        solvers_body = preflight.block_body(solution_text, "solvers")
        for field, body in preflight.block_entries(solvers_body):
            values = preflight.entry_values(body)
            bits = [f"{key} {values[key]}" for key in ("solver", "smoother", "tolerance",
                                                       "relTol", "preconditioner")
                    if key in values]
            lines.append(f"  {field:<12} " + ", ".join(bits))
        for block in ("SIMPLE", "PIMPLE", "PISO"):
            body = preflight.block_body(solution_text, block)
            if not body:
                continue
            lines.append(f"  {block}")
            for key, value in preflight.entry_values(preflight.outer_text(body)).items():
                if value.strip():
                    lines.append(f"    {key:<24} {value}")
            control_body = preflight.block_body(body, "residualControl")
            if control_body.strip():
                lines.append("    residualControl")
                lines.extend(f"      {line}" for line in as_written(control_body))
        relax = preflight.block_body(solution_text, "relaxationFactors")
        if relax.strip():
            lines.append("  relaxationFactors")
            lines.extend(f"    {line}" for line in as_written(relax))
    else:
        lines.append("linear solvers and convergence: system/fvSolution was not readable")
    lines.append("")

    zero = (case / "0") if case is not None else None
    fields = sorted(p.name for p in zero.iterdir() if p.is_file()) \
        if zero is not None and zero.is_dir() else []
    if fields:
        read.append("0/")
        lines.append(f"initial and boundary conditions (0/: {', '.join(fields)})")
        for field in fields:
            lines.append(f"  {field}  {field_shape(read_dict(case, f'0/{field}'))}")
            patches, unread = boundary_types(case, field)
            for patch, patch_type in patches:
                lines.append(f"    {patch:<22} {patch_type}")
            if unread:
                lines.append(f"    {'(not read)':<22} {unread}")
    else:
        lines.append("initial and boundary conditions: no 0/ directory in the case")
    lines.append("")

    check = find_check_log(case)
    if check is not None:
        read.append(check.name)
        data = mesh_digest.parse(check.read_text(errors="replace"))
        lines.append(f"mesh ({check.name})")
        for key, value in sorted(data.get("counts", {}).items()):
            lines.append(f"  {key:<16} {value:,}")
        for key in ("non_ortho", "skewness", "aspect_ratio", "openness",
                    "face_determinant", "cell_volume", "face_area", "bounding_box"):
            if key in data:
                value = data[key]
                lines.append(f"  {key:<16} "
                             + (" to ".join(str(v) for v in value)
                                if isinstance(value, tuple) else str(value)))
        if data.get("patches"):
            lines.append("  patches")
            for row in data["patches"]:
                lines.append(f"    {row[0]:<22} {row[1]} faces, {row[2]} points")
        for failure in data.get("failures", []):
            lines.append(f"  FAILED: {failure}")
        for warning in data.get("warnings", []):
            lines.append(f"  warning: {warning}")
        if data.get("mesh_ok_line"):
            lines.append(f"  {data['mesh_ok_line']}")
    else:
        lines.append("mesh: no checkMesh output in the case")
    lines.append("")

    log = results.find_solver_log(case) if case is not None else None
    if log is not None:
        read.append(log.name)
        data = log_digest.digest(log)
        lines.append(f"what the solver did ({log.name})")
        for line in log_digest.how_it_ended(data, log_digest.requested_end_time(log)):
            lines.extend(f"  {wrapped}" for wrapped in _wrap(line, 96, "    "))
        times = data.get("times") or []
        if times:
            axis = "last iteration" if is_steady_solver(
                str(control.get("application", ""))) else "last t ="
            lines.append(f"  steps parsed {len(times)}, {axis} {times[-1]:g}")
    else:
        lines.append("what the solver did: no solver log in the case")
    lines.append("")

    if layers and case is not None and (case / "constant" / "polyMesh" / "points").exists():
        import layer_report  # noqa: PLC0415 - reads the whole mesh; only when asked

        measured = layer_report.measure(case)
        lines.append("wall layers (layer_report.py, from constant/polyMesh)")
        for patch in measured.get("patches", []):
            first = patch.get("first_cell_m") or {}
            median = first.get("median")
            lines.append(f"  {str(patch.get('patch', '?')):<22} "
                         f"type {patch.get('type', '?')}, "
                         f"faces {patch.get('faces', '?')}, "
                         "first cell height median "
                         + (f"{median:.4g} m" if isinstance(median, float) else "not measured")
                         + (f"  {patch['note']}" if patch.get("note") else ""))
        for problem in measured.get("problems", []):
            lines.append(f"  not measured: {problem}")
        read.append("constant/polyMesh")
    elif layers:
        lines.append("wall layers: no constant/polyMesh/points to measure")
    else:
        lines.append("wall layers: not measured (--no-layers)")

    return [_capped(line) for line in lines], read


def _capped(line: str) -> str:
    """One line of the methods section, at most `MAX_VALUE_CHARS` long.

    Applied here, to every line, rather than at each of the fifteen places a value is
    read: #139 was one unbounded value from one case file reaching the page, and the
    next one will come from a dictionary nobody has thought about yet. A line that was
    cut says so, because a reader who cannot see that something was left out cannot go
    and read the file.
    """
    if len(line) <= MAX_VALUE_CHARS:
        return line
    return (f"{line[:MAX_VALUE_CHARS]} ... -- summarised, not printed "
            f"({len(line) - MAX_VALUE_CHARS:,} more characters)")


# -- what was asked for and not produced ----------------------------------------------


def gaps(study: Path, case: str = "") -> list[str]:
    """What the study itself recorded as skipped or failed, in its own words.

    Read rather than inferred. `phases.json` distinguishes `skipped` from `pending` on
    purpose -- "not wanted" and "not done" do not read the same -- and `results.py`
    carries each skipped output's reason into `results.md` because "there is no forces
    plot" and "there is no forces plot because nothing logged any" are different facts.
    Both are quoted here; nothing new is concluded.
    """
    lines: list[str] = []
    phases = study_state.load_phases(study)
    rows = [row for row in phases.get("phases", [])
            if isinstance(row, dict) and row.get("status") in ("skipped", "failed")]
    if rows:
        lines.append("phases the study did not complete (.reynolds/phases.json)")
        for row in rows:
            note = str(row.get("note") or "")
            lines.append(f"  {str(row.get('name')):<12} {row.get('status')}"
                         + (f"  {note}" if note else ""))
        lines.append("")

    latest = study_state.latest(REPORT_KIND, root=study, case=case)
    if latest and str(latest.get("path", "")).endswith(".md"):
        path = Path(str(latest["abspath"]))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for heading in ("## skipped", "## failed"):
            if heading not in text:
                continue
            block = text.split(heading, 1)[1].split("\n## ", 1)[0]
            quoted = [line.strip() for line in block.splitlines() if line.strip().startswith("-")]
            if quoted:
                lines.append(f"{heading.strip('# ')}, as {path.name} recorded it")
                lines.extend(f"  {line}" for line in quoted)
                lines.append("")
    if not lines:
        lines.append("The study recorded nothing as skipped or failed. That is the "
                     "record, not a guarantee that nothing was left out.")
    return lines


# -- drawing the pages ----------------------------------------------------------------


def _wrap(text: str, width: int, indent: str = "  ") -> list[str]:
    """Hand-rolled because every line here is already laid out for a monospaced page,
    and `textwrap` would re-flow the alignment out of the tables above.

    A word longer than the width is cut rather than left to run off the edge of the
    page, because the words that do that here are absolute paths and a path that runs
    into the margin is a path nobody can read.
    """
    width = max(width, len(indent) + 8)
    out: list[str] = []
    line = ""
    for word in str(text).split():
        while len(indent) + len(word) > width:
            if line.strip() and width - len(line) - 1 < 8:
                # Not worth starting a path on the tail of a line.
                out.append(line)
                line = ""
                continue
            room = (width - len(line) - 1) if line.strip() else width
            out.append(f"{line} {word[:room]}".rstrip() if line.strip() else word[:room])
            word = word[room:]
            line = indent
        candidate = f"{line} {word}".rstrip() if line.strip() else f"{line}{word}"
        if len(candidate) > width and line.strip():
            out.append(line)
            line = indent + word
        else:
            line = candidate
    if line.strip():
        out.append(line)
    return out or [""]


def _blank_page() -> tuple[Any, Any]:
    figure = plt.figure(figsize=PAGE_INCHES, dpi=PAGE_DPI)
    figure.patch.set_facecolor("white")
    return figure, figure.add_axes((0, 0, 1, 1))


def _frame(axes) -> None:
    axes.set_axis_off()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)


def place_logo(figure, logo: Path | None, notes: list[str]) -> None:
    """The logo on the cover, at whatever aspect ratio it came in at.

    The box is fixed and the image keeps its own shape inside it, which is the one
    thing a logo must not lose: a wordmark stretched to fill a square is worse than no
    logo. A file that will not read back is a note, not an exception -- the document is
    the deliverable and the logo is decoration on it.
    """
    if logo is None:
        return
    try:
        image = mpimg.imread(str(logo))
    except Exception as exc:  # noqa: BLE001
        notes.append(f"logo not drawn: {type(exc).__name__}: {exc}")
        return
    height, width = image.shape[0], image.shape[1]
    box_width, box_height = 0.20, 0.07
    scale = min(box_width / max(width, 1), box_height / max(height, 1))
    draw_width, draw_height = width * scale, height * scale
    # Right-aligned in the box, so a wide wordmark and a square mark both end at the
    # same margin. The box is in figure coordinates; the image keeps its own aspect.
    left = 0.93 - draw_width
    axes = figure.add_axes((left, 0.935 - draw_height, draw_width, draw_height))
    axes.imshow(image)
    axes.set_axis_off()


def cover_page(
    pdf: PdfPages,
    *,
    template: dict[str, Any],
    study: Path,
    digest: dict[str, Any],
    numbers: list[dict[str, Any]],
    hero: Path | None,
    hero_rule: str,
    logo: Path | None,
    notes: list[str],
) -> None:
    """The one page somebody who reads nothing else will read: who this is, what the
    numbers are, whether each has a figure behind it, and one picture."""
    figure, axes = _blank_page()
    _frame(axes)
    place_logo(figure, logo, notes)

    title = str(template.get("title") or f"{digest['study']} -- study report")
    axes.text(0.07, 0.945, title, fontsize=19, va="top", ha="left", weight="bold")
    subtitle = str(template.get("subtitle") or "")
    y = 0.905
    if subtitle:
        axes.text(0.07, y, subtitle, fontsize=10.5, va="top", ha="left", color="#44555f")
        y -= 0.028

    identity = [
        f"study        {digest['study']}  ({study})",
        f"case         {digest.get('case') or '-'}",
        "phases       " + "  ".join(f"{name}:{status}" for name, status in digest["phases"]),
        f"artifacts    {digest['artifact_count']} registered and on disk",
        f"written      {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
    ]
    for line in identity:
        for wrapped in _wrap(line, 100, "             "):
            axes.text(0.07, y, wrapped, fontsize=7.4, va="top", ha="left",
                      family="monospace", color="#2b3b45")
            y -= LINE_PITCH
    for line in template.get("front_matter") or []:
        for wrapped in _wrap(str(line), 100, "  "):
            axes.text(0.07, y, wrapped, fontsize=7.6, va="top", ha="left", color="#2b3b45")
            y -= LINE_PITCH
    y -= 0.012

    axes.text(0.07, y, "key numbers", fontsize=11.5, va="top", ha="left", weight="bold")
    y -= 0.026
    axes.text(0.07, y,
              "Each number is shown with where it came from: the figure it is stamped "
              "into, or the file it was read out of.",
              fontsize=7.6, va="top", ha="left", color="#55666f")
    y -= 0.016
    for wrapped in _wrap(stamp_backing(numbers), 118, ""):
        axes.text(0.07, y, wrapped, fontsize=7.6, va="top", ha="left", color="#55666f")
        y -= 0.013
    y -= 0.011

    colour = {"verified": "#1b5e20", "disagrees": "#b71c1c", "unverified": "#5c6f7d"}
    # The state column only earns its width when the states differ. Nothing in this
    # pipeline stamps a figure, so on a real study every row reads `unverified`: a
    # column repeating one word down the page reads as a verdict on each number, when
    # it is one fact about all of them and `stamp_backing` above has already said it.
    varied = len({row["state"] for row in numbers}) > 1
    if not numbers:
        axes.text(0.07, y, "none: this study has recorded no claim and its files carry "
                           "no coefficient or residual",
                  fontsize=8, va="top", ha="left", family="monospace", color="#5c6f7d")
        y -= LINE_PITCH
    for row in numbers:
        state = row["state"]
        if varied:
            axes.text(0.07, y, f"{state:<11}", fontsize=7.6, va="top", ha="left",
                      family="monospace", color=colour.get(state, "#5c6f7d"),
                      weight="bold")
        axes.text(0.185 if varied else 0.07, y, f"{row['label']}: {row['text']}",
                  fontsize=7.6, va="top", ha="left", family="monospace")
        y -= LINE_PITCH
        for wrapped in _wrap(row["why"], 82, ""):
            axes.text(0.195 if varied else 0.08, y, wrapped, fontsize=6.9,
                      va="top", ha="left",
                      family="monospace", color="#55666f")
            y -= LINE_PITCH * 0.92
        if y < 0.44:
            axes.text(0.07, y, "... the rest are in the sidecar and on stdout",
                      fontsize=7, va="top", ha="left", family="monospace", color="#5c6f7d")
            break

    hero_box = figure.add_axes((0.07, 0.07, 0.86, 0.30))
    hero_box.set_xticks([])
    hero_box.set_yticks([])
    drawn = False
    if hero is not None:
        try:
            hero_box.imshow(mpimg.imread(str(hero)))
            drawn = True
        except Exception as exc:  # noqa: BLE001
            hero_box.text(0.5, 0.5, f"hero figure could not be read back\n{hero}\n"
                                    f"{type(exc).__name__}: {exc}",
                          transform=hero_box.transAxes, ha="center", va="center",
                          family="monospace", fontsize=7.5, color="#78909c")
            notes.append(f"hero not drawn: {type(exc).__name__}: {exc}")
    else:
        hero_box.text(0.5, 0.5, "no hero figure", transform=hero_box.transAxes,
                      ha="center", va="center", family="monospace", fontsize=8,
                      color="#78909c")
    # The name and the rule, not the whole path: the path is in the sidecar and on
    # stdout, and a caption that runs off into the margin is a caption nobody reads.
    caption = [f"hero: {hero.name if hero else 'none'}"]
    caption += _wrap(f"chosen because: {hero_rule}", 112, "")
    hero_box.set_title("\n".join(caption), fontsize=6.8, loc="left", color="#44555f")
    if not drawn:
        hero_box.set_facecolor("#fafafa")

    footer = str(template.get("footer") or "")
    if footer:
        for index, wrapped in enumerate(_wrap(footer, 110, "")):
            axes.text(0.07, 0.045 - index * 0.016, wrapped, fontsize=6.6, va="top",
                      ha="left", color="#55666f")
    pdf.savefig(figure)
    plt.close(figure)


def stamp_backing(numbers: list[dict[str, Any]]) -> str:
    """What this page can and cannot say about where its numbers came from.

    The sweep behind #139 and #145 found that not one PNG in `studies/` or
    `data/qa-runs/` carries a `reynolds-claims` stamp -- re-measured over 973 PNGs,
    still zero -- because nothing that draws a figure writes one. `claims.attach` is
    the only call that stamps, and no renderer calls it.

    So `verified` is not an outcome this page can reach on a real study, and the first
    version of this line said so as a count: `0 of 22`. A count of zero reads as a
    check that was run and found nothing, which is the opposite of the truth -- nothing
    was ever stamped to check against. It is the measurement that is missing, not the
    agreement. So where nothing is stamped this says what it knows in words and offers
    no tally, and the tally comes back only when a stamp does.
    """
    if not numbers:
        return "no number reached this page"
    backed = sum(1 for row in numbers if row.get("state") == "verified")
    if not backed:
        return ("no figure here is stamped with a value -- nothing in this pipeline "
                "writes the reynolds-claims stamp -- so there was nothing for these "
                "numbers to be checked against. Each one is read out of the study's "
                "own files and is shown with the file it came from")
    return (f"{backed} of {len(numbers)} are backed by a figure stamped with the "
            "value; the rest are not")


BODY_WIDTH = 100
"""Monospaced characters that fit across a text page at `BODY_SIZE`. Lines are folded
to it here rather than where they are built, so one page-geometry number governs every
section and no section has to know how wide the page is."""


def text_pages(
    pdf: PdfPages,
    title: str,
    lines: Iterable[str],
    *,
    max_pages: int | None = None,
    notes: list[str] | None = None,
) -> int:
    """A block of already-laid-out monospaced lines, paginated. Returns the pages.

    `max_pages` is #139's other guard. The per-value cap stops one field filling a
    document, and this stops the section doing it: a `0/` with forty fields in it, or
    a dictionary nobody has met yet, still cannot turn a four-page report into a
    thousand-page one. The cut is stated on the last page it writes and in `notes`,
    because a section that stops early without saying so reads as a section that
    ended.
    """
    folded: list[str] = []
    for line in lines:
        if len(line) <= BODY_WIDTH:
            folded.append(line)
            continue
        # The leading space is the table column the line sits in, so it is kept and
        # the fold happens inside it.
        pad = line[:len(line) - len(line.lstrip())]
        folded.extend(pad + piece
                      for piece in _wrap(line.strip(), BODY_WIDTH - len(pad), "    "))
    lines = folded or ["(nothing)"]
    if max_pages is not None and len(lines) > max_pages * BODY_LINES:
        room = max_pages * BODY_LINES - 2
        cut = (f"-- {len(lines) - room:,} further line(s) are not printed: this "
               f"section is capped at {max_pages} pages.")
        if notes is not None:
            notes.append(f"{title}: {cut.lstrip('- ')}")
        lines = lines[:room] + ["", cut]
    pages = 0
    for start in range(0, len(lines), BODY_LINES):
        chunk = lines[start:start + BODY_LINES]
        figure, axes = _blank_page()
        _frame(axes)
        heading = title if start == 0 else f"{title} (continued)"
        axes.text(0.07, 0.955, heading, fontsize=13, va="top", ha="left", weight="bold")
        y = 0.915
        for line in chunk:
            axes.text(0.07, y, line, fontsize=BODY_SIZE, va="top", ha="left",
                      family="monospace")
            y -= LINE_PITCH
        pdf.savefig(figure)
        plt.close(figure)
        pages += 1
    return pages


def figure_pages(pdf: PdfPages, groups: list[tuple[str, list[gallery.Entry]]]) -> tuple[int, list[dict]]:
    """The figures, grouped as the gallery groups them, four to a page.

    A row whose file is gone gets its panel anyway, labelled MISSING. `gallery.py` shows
    it that way for a reason worth repeating here: a manifest saying a picture once
    existed is often the only surviving evidence that a step ran, and a document that
    drops it makes a deleted figure look like one that was never made.
    """
    pages = 0
    placed: list[dict] = []
    for kind, group in groups:
        pick = gallery.newest(group)
        for start in range(0, len(group), FIGURES_PER_PAGE):
            chunk = group[start:start + FIGURES_PER_PAGE]
            figure = plt.figure(figsize=PAGE_INCHES, dpi=PAGE_DPI)
            figure.patch.set_facecolor("white")
            gone = sum(1 for entry in group if not entry.exists)
            heading = f"{kind}  ({len(group)}" + (f", {gone} missing" if gone else "") + ")"
            figure.suptitle(heading, fontsize=12, x=0.07, ha="left", weight="bold")
            # Sized to what is on the page, through the gallery's own rule, so a kind
            # with one picture gets a page with one picture on it rather than a
            # quarter-page panel and three empty frames.
            rows, cols = gallery.grid_shape(len(chunk), max_cols=2)
            grid = figure.subplots(rows, cols, squeeze=False)
            cells = [cell for row in grid for cell in row]
            for cell in cells[len(chunk):]:
                cell.set_axis_off()
            for cell, entry in zip(cells, chunk):
                cell.set_xticks([])
                cell.set_yticks([])
                for spine in cell.spines.values():
                    spine.set_color("#b0bec5")
                    spine.set_linewidth(0.8)
                state = "MISSING" if not entry.exists else ""
                picture = gallery.image_for(entry)
                if not entry.exists:
                    cell.set_facecolor("#fbfbfb")
                    cell.text(0.5, 0.5, "MISSING\nrecorded, not on disk now\n"
                                        f"{entry.rel}",
                              transform=cell.transAxes, ha="center", va="center",
                              family="monospace", fontsize=7, color="#90a4ae")
                elif picture is None:
                    cell.set_facecolor("#fbfbfb")
                    cell.text(0.5, 0.5, f"not a picture\n{entry.rel}",
                              transform=cell.transAxes, ha="center", va="center",
                              family="monospace", fontsize=7, color="#90a4ae")
                    state = "not a picture"
                else:
                    try:
                        cell.imshow(mpimg.imread(str(picture)))
                    except Exception as exc:  # noqa: BLE001 - one panel, not one page
                        cell.set_facecolor("#fbfbfb")
                        cell.text(0.5, 0.5, f"{picture.name}\ncould not be read back\n"
                                            f"{type(exc).__name__}: {exc}",
                                  transform=cell.transAxes, ha="center", va="center",
                                  family="monospace", fontsize=6.5, color="#90a4ae")
                        state = "unreadable"
                label = gallery.describe(entry) or entry.rel
                mark = "  [latest]" if entry is pick and len(group) > 1 else ""
                cell.set_title("\n".join(_wrap(f"{entry.rel}{mark}", 52, "  ")
                                         + _wrap(label, 52, "  ")),
                               fontsize=6.4, loc="left", color="#44555f")
                placed.append({
                    "kind": kind, "path": str(entry.path), "rel": entry.rel,
                    "exists": entry.exists, "state": state or "drawn",
                    "page": pages + 1,
                })
            figure.tight_layout(rect=(0.02, 0.01, 0.98, 0.95))
            pdf.savefig(figure)
            plt.close(figure)
            pages += 1
    return pages, placed


# -- the whole document ----------------------------------------------------------------


def write_report(
    given: Path,
    out: Path,
    *,
    template: dict[str, Any],
    case: str = "",
    hero: Path | None = None,
    logo: Path | None = None,
    layers: bool = True,
) -> dict[str, Any]:
    """The document, and a record of what went into it and what was missing."""
    where = locate(given)
    study, bases, base = where["state_root"], where["bases"], where["base"]
    rows = manifest_rows(study, bases, case=case) if where["is_study"] else []
    entries = [gallery.entry_from_row(row) for row in rows
               if not gallery.is_own_output(row)
               and (row.get("meta") or {}).get("source") != SOURCE_TAG]
    groups = gallery.group_by_kind(entries)
    figure_claims = stamped_claims(study, bases, base, case=case)
    case_dir, case_rule, others = choose_case(study, case)
    measured, measure_notes = measured_numbers(case_dir)
    numbers = tie_numbers(figure_claims, measured)
    # A number that is zero to within the record's own scale is not a headline, so it
    # goes under the ones that are. Stable, so nothing else is reordered.
    numbers.sort(key=lambda row: bool(row.get("negligible")))
    hero_path, hero_rule = pick_hero(entries, figure_claims, hero)
    notes = where["notes"] + list(measure_notes)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pages: list[str] = []
    placed: list[dict] = []
    methods: list[str] = []
    sources: list[str] = []
    gap_lines: list[str] = []

    with PdfPages(str(out)) as pdf:
        for section in template["sections"]:
            if section == "summary":
                digest = study_state.digest(study)
                if base != study:
                    digest["study"] = base.name
                cover_page(
                    pdf, template=template, study=study,
                    digest=digest, numbers=numbers,
                    hero=hero_path, hero_rule=hero_rule, logo=logo, notes=notes,
                )
                pages.append("summary")
            elif section == "figures":
                count, placed = figure_pages(pdf, groups)
                if count == 0:
                    count = text_pages(pdf, "figures", [
                        "No artifact is registered in this study's manifest.",
                        f"The manifest is {study / study_state.STATE_DIR / study_state.MANIFEST}.",
                    ])
                pages.extend(["figures"] * count)
            elif section == "methods":
                methods, sources = methods_lines(case_dir, case_rule, others, layers=layers)
                pages.extend(["methods"] * text_pages(
                    pdf, "methods", methods,
                    max_pages=METHODS_MAX_PAGES, notes=notes))
            elif section == "gaps":
                gap_lines = gaps(study, case=case)
                pages.extend(["gaps"] * text_pages(
                    pdf, "asked for and not produced", gap_lines))

    return {
        "study": str(study),
        "given": str(given),
        "state_root": str(study),
        "artifact_base": str(base),
        "is_study": where["is_study"],
        "stamp_backing": stamp_backing(numbers),
        "out": str(out.resolve()),
        "pages": len(pages),
        "page_sections": pages,
        "case": str(case_dir) if case_dir else "",
        "case_rule": case_rule,
        "other_cases": others,
        "hero": str(hero_path) if hero_path else "",
        "hero_rule": hero_rule,
        "logo": str(logo) if logo else "",
        "numbers": numbers,
        "figures": placed,
        "missing": [row for row in placed if not row["exists"]],
        "methods_read": sources,
        "gaps": gap_lines,
        "notes": notes,
        "sections": list(template["sections"]),
    }


def render_summary(record: dict[str, Any]) -> str:
    """The same answer in the terminal, because most of the time that is enough."""
    lines = [
        f"{record['out']}",
        f"{record['pages']} page(s): " + ", ".join(record["page_sections"]),
        f"hero: {record['hero'] or 'none'}",
        f"  rule: {record['hero_rule']}",
        f"case for the methods section: {record['case'] or 'none'}  ({record['case_rule']})",
    ]
    counts: dict[str, int] = {}
    for row in record["numbers"]:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    lines.append("key numbers: " + (", ".join(f"{state} x{count}"
                                              for state, count in sorted(counts.items()))
                                    or "none"))
    lines.append(f"stamp-backed: {record['stamp_backing']}")
    for row in record["numbers"]:
        if row["state"] != "verified":
            lines.append(f"  {row['state']}: {row['label']}: {row['text']} -- {row['why']}")
    if record["missing"]:
        lines.append(f"{len(record['missing'])} registered artifact(s) are MISSING and are "
                     "on the pages as MISSING:")
        for row in record["missing"]:
            lines.append(f"  {row['rel']}")
    if record["methods_read"]:
        lines.append("methods read: " + ", ".join(record["methods_read"]))
    for note in record["notes"]:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", nargs="?", default=Path("."), type=Path,
                        help="a path inside the study (default: here)")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"where the PDF goes (default: <study>/{PDF_NAME}); a "
                             "relative path is relative to the directory you run this "
                             "from, not to the case")
    parser.add_argument("--json", dest="sidecar", type=Path, default=None,
                        help="write what went into the document, and what was missing,"
                             " as JSON")
    parser.add_argument("--template", type=Path, default=None,
                        help=f"a JSON template: title, subtitle, front_matter, sections,"
                             f" footer (example: {TEMPLATE_NAME})")
    parser.add_argument("--logo", type=Path, default=None,
                        help="an image for the top right of the cover")
    parser.add_argument("--hero", type=Path, default=None,
                        help="the figure for the cover, instead of the chosen one")
    parser.add_argument("--case", default="",
                        help="the case the methods section describes, and the only"
                             " case's artifacts to include")
    parser.add_argument("--no-layers", dest="layers", action="store_false",
                        help="skip the wall-layer measurement, which reads the whole"
                             " polyMesh")
    args = parser.parse_args(argv)

    # One handler for every `Refused`, not just the template's (#186): a `--case` that
    # names nothing and a `root` that is not there are the same kind of mistake, and
    # the exit code a caller checks should not depend on which argument was wrong.
    try:
        template, template_notes = load_template(args.template)
        record = write_report(
            args.root, args.out or study_state.find_root(args.root) / PDF_NAME,
            template=template, case=args.case, hero=args.hero, logo=args.logo,
            layers=args.layers,
        )
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    record["template"] = str(args.template) if args.template else "(built in)"
    record["notes"] = template_notes + record["notes"]

    # #145: pointed at a study top, this used to write a second
    # `.reynolds/manifest.jsonl` beside the real one and split the study's state in
    # two. A directory with no `.reynolds/` under it is not a study, and the document
    # is still written -- what is refused is inventing a study to register it in.
    if record["is_study"]:
        study_state.record(
            REPORT_KIND, record["out"], root=Path(record["state_root"]), case=args.case,
            label=f"report pack, {record['pages']} pages", source=SOURCE_TAG,
        )
        record["registered_as"] = REPORT_KIND
    else:
        record["registered_as"] = ""

    print(render_summary(record))
    if record["registered_as"]:
        print(f"registered in the manifest under kind '{REPORT_KIND}' in "
              f"{record['state_root']}")
    else:
        print(f"not registered: nothing under {args.root} has a "
              f"{study_state.STATE_DIR}/, so it is not a study, and writing one here "
              "would give it a second state directory")
    if args.sidecar:
        Path(args.sidecar).parent.mkdir(parents=True, exist_ok=True)
        Path(args.sidecar).write_text(json.dumps(record, indent=2, default=str),
                                      encoding="utf-8")
        print(Path(args.sidecar).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
